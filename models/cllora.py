import logging
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import OurNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.shared_adapter_ewc import SharedAdapterEWC
import random

num_workers = 8


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def compute_orthogonality_loss(previous_weights_list, current_weights, epsilon=1e-8):
    total_ortho_loss = current_weights.new_zeros(())
    current_norm = torch.norm(current_weights.flatten())
    current_normalized = current_weights.flatten() / (current_norm + epsilon)

    for prev_weights in previous_weights_list:
        # Normalize previous weights
        prev_norm = torch.norm(prev_weights.flatten())
        prev_normalized = prev_weights.flatten() / (prev_norm + epsilon)

        # Compute absolute dot product (should be close to 0 for orthogonal vectors)
        dot_product = torch.abs(torch.sum(prev_normalized * current_normalized))

        total_ortho_loss += dot_product

    # Average over all previous tasks
    if len(previous_weights_list) > 0:
        total_ortho_loss /= len(previous_weights_list)

    return total_ortho_loss


def compute_optional_orthogonality_loss(previous_weights_list, current_weights, use_orthogonal_constraint, epsilon=1e-8):
    if current_weights is None:
        if previous_weights_list:
            return previous_weights_list[0].detach().new_zeros(())
        return torch.zeros(())
    if not use_orthogonal_constraint:
        return current_weights.detach().new_zeros(())
    if len(previous_weights_list) == 0:
        return current_weights.new_zeros(())
    return compute_orthogonality_loss(previous_weights_list, current_weights, epsilon=epsilon)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = OurNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]

        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"] # forward_reweight is divide by _cur_task
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]
        # use_orth_loss only controls the orthogonality loss; it is fully decoupled
        # from use_block_weight (block-wise weight). Prefer the nested
        # task_specific_adapter block, fall back to the legacy flat key.
        tsa_cfg = args.get("task_specific_adapter", {}) or {}
        self.use_orthogonal_constraint = bool(
            tsa_cfg.get("use_orth_loss", args.get("use_orthogonal_constraint", False))
        )
        self.orthogonal_lambda = float(args.get("orthogonal_lambda", 0.0))
        self.args["use_orthogonal_constraint"] = self.use_orthogonal_constraint
        self.args["orthogonal_lambda"] = self.orthogonal_lambda
        # B5: orthogonality between the current SD-LoRA direction and the frozen
        # old directions (the actual task subspaces), as opposed to the legacy
        # block_weight orthogonality above. Off unless direction_orth_lambda > 0.
        self.direction_orth_lambda = float(tsa_cfg.get("direction_orth_lambda", 0.0))
        logging.info(
            "Orthogonal constraint: enabled=%s lambda=%s direction_orth_lambda=%s",
            self.use_orthogonal_constraint,
            self.orthogonal_lambda,
            self.direction_orth_lambda,
        )

        # EWC regularization on the CL-LoRA shared (general_pos) adapters only.
        # Fully decoupled from SD-LoRA-RR, which only touches task-specific
        # (specfic_pos) adapters.
        self.ewc = SharedAdapterEWC(args)
        if self.ewc.enable:
            self.ewc.log_config()

        # Branch score calibration (B1): make the per-branch cosine logits of
        # forward_diagonal comparable across task branches. Stats are collected
        # for free inside replace_fc (which already forwards the current task
        # data through every branch) and applied as a per-branch affine map on
        # the concatenated logits at eval time. Never touches training.
        # B8: prototypes from 1 clean pass + (proto_views-1) augmented passes
        # over the task data; ridge/calibration stats see the same views.
        self.proto_views = int(args.get("proto_views", 1))

        calib_cfg = args.get("branch_calibration", {}) or {}
        self.calib_enable = bool(calib_cfg.get("enable", False))
        self.calib_scheme = calib_cfg.get("scheme", "pos_zscore")
        self.calib_neg_momentum = float(calib_cfg.get("neg_momentum", 0.5))
        self.calib_eps = float(calib_cfg.get("eps", 1e-6))
        self.dump_eval = bool(calib_cfg.get("dump_eval", False))
        # branch -> (mu, sigma) of the max cosine score of that branch's own
        # diagonal block. pos: on the branch's own task data (frozen once).
        # neg: on later tasks' data, i.e. out-of-task samples (momentum-updated).
        self.calib_pos_stats = {}
        self.calib_neg_stats = {}
        # B3: online ridge head in the (frozen after task 0) branch-0 feature
        # space. Rehearsal-free: only the Gram matrix G [d,d] and target matrix
        # C [d, n_classes] are accumulated, never any sample. At eval the closed
        # form W = (G + lambda I)^-1 C scores all classes in one shared space,
        # and is (optionally) ensembled with the calibrated diagonal logits.
        ridge_cfg = args.get("ridge_head", {}) or {}
        self.ridge_enable = bool(ridge_cfg.get("enable", False))
        self.ridge_lambda = float(ridge_cfg.get("lambda", 1.0))
        self.ridge_weight = float(ridge_cfg.get("ensemble_weight", 0.2))
        self.ridge_mode = ridge_cfg.get("mode", "ensemble")
        self.ridge_G = None
        self.ridge_C = None
        if self.ridge_enable:
            logging.info(
                "Ridge head: enable=%s lambda=%s mode=%s ensemble_weight=%s",
                self.ridge_enable,
                self.ridge_lambda,
                self.ridge_mode,
                self.ridge_weight,
            )
        if self.calib_enable or self.dump_eval:
            logging.info(
                "Branch calibration: enable=%s scheme=%s neg_momentum=%s dump_eval=%s",
                self.calib_enable,
                self.calib_scheme,
                self.calib_neg_momentum,
                self.dump_eval,
            )

        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1
            self.beta = 1

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()
        # Per-task tensor sizes grow monotonically (direction caches, graph
        # intermediates scale with task count), so freed blocks from task t-1
        # never fit task t's allocations and the caching allocator keeps
        # opening new segments: reserved memory hit 21.9GB at 40 tasks while
        # allocated peaked at 7GB. Releasing cached segments once per task
        # bounds reserved at the true transient peak (~9.4GB at 40 tasks).
        torch.cuda.empty_cache()

    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc

        return start_cls, end_cls

    def replace_fc_proxy(self):
        model = self._network
        model = model.eval()
        model.fc.weight.data[self._known_classes:self._total_classes, :] = model.proxy_fc.weight.data
        model.fc.bias.data[self._known_classes:self._total_classes] = model.proxy_fc.bias.data

    def replace_fc(self, train_loader):
        model = self._network
        model = model.eval()

        with torch.no_grad():
            # replace proto for each adapter in the current task
            if self.use_init_ptm:
                start_idx = -1
            else:
                start_idx = 0

            for index in range(start_idx, self._cur_task + 1):
                if self.moni_adam:
                    if index > self.adapter_num - 1:
                        break
                # only use the diagonal feature, index = -1 denotes using init PTM, index = self._cur_task denotes the last adapter's feature
                elif self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                loaders = [train_loader]
                if self.proto_views > 1:
                    # extra passes over the augmented (mode="train") dataset;
                    # each pass samples fresh augmentations
                    aug_loader = DataLoader(
                        self.train_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                    )
                    loaders += [aug_loader] * (self.proto_views - 1)

                embedding_list, label_list = [], []
                for loader in loaders:
                    for i, batch in enumerate(loader):
                        (_, data, label) = batch
                        data = data.to(self._device)
                        label = label.to(self._device)
                        embedding = model.backbone.forward_proto(data, adapt_index=index)
                        embedding_list.append(embedding.cpu())
                        label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                class_list = np.unique(self.train_dataset_for_protonet.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto

                # B1: collect per-branch calibration stats (and optional dumps)
                # from the embeddings this loop already computed.
                if (self.calib_enable or self.dump_eval) and index >= 0:
                    self._collect_branch_stats(index, embedding_list, label_list)
                # B3: accumulate ridge statistics in the branch-0 space.
                if self.ridge_enable and index == 0:
                    self._ridge_accumulate(embedding_list, label_list)
        return

    def _ridge_accumulate(self, embedding_list, label_list):
        x = F.normalize(embedding_list.float(), p=2, dim=1)
        gram = x.t() @ x
        self.ridge_G = gram if self.ridge_G is None else self.ridge_G + gram
        if self.ridge_C is None:
            self.ridge_C = torch.zeros(x.shape[1], self._total_classes)
        elif self.ridge_C.shape[1] < self._total_classes:
            expanded = torch.zeros(x.shape[1], self._total_classes)
            expanded[:, : self.ridge_C.shape[1]] = self.ridge_C
            self.ridge_C = expanded
        onehot = torch.zeros(x.shape[0], self._total_classes)
        onehot[torch.arange(x.shape[0]), label_list.long()] = 1.0
        self.ridge_C += x.t() @ onehot
        logging.info(
            "[Ridge Head] accumulated task=%s samples=%s total_classes=%s",
            self._cur_task,
            x.shape[0],
            self.ridge_C.shape[1],
        )

    def _ridge_solve(self):
        if self.ridge_G is None or self.ridge_C is None:
            return None
        dim = self.ridge_G.shape[0]
        gram = self.ridge_G + self.ridge_lambda * torch.eye(dim)
        return torch.linalg.solve(gram, self.ridge_C)

    @staticmethod
    def _zscore(matrix, eps=1e-8):
        return (matrix - matrix.mean()) / (matrix.std() + eps)

    def _analysis_dir(self):
        init_cls = 0 if self.args["init_cls"] == self.args["increment"] else self.args["init_cls"]
        path = "logs/{}/{}/{}/{}/analysis_{}_{}".format(
            self.args["model_name"],
            self.args["dataset"],
            init_cls,
            self.args["increment"],
            self.args["prefix"],
            self.args["seed"],
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _collect_branch_stats(self, index, embedding_list, label_list):
        """Update calibration stats for branch `index` from current-task embeddings.

        embedding_list: [N, out_dim] cpu features of the CURRENT task's data under
        branch `index` (computed by replace_fc). The score used everywhere is the
        max cosine similarity against the branch's own diagonal prototypes, i.e.
        exactly what forward_diagonal compares across branches at test time."""
        out_dim = self._network.out_dim
        start_cls, end_cls = self.get_cls_range(index)
        start_dim = index * out_dim
        protos = self._network.fc.weight.data[start_cls:end_cls, start_dim:start_dim + out_dim].cpu()
        emb = F.normalize(embedding_list.float(), p=2, dim=1)
        protos = F.normalize(protos.float(), p=2, dim=1)
        max_scores = (emb @ protos.t()).max(dim=1).values
        mu = max_scores.mean().item()
        sigma = max_scores.std().item()

        if index == self._cur_task:
            # current task's own branch -> positive statistics, frozen afterwards
            self.calib_pos_stats[index] = (mu, sigma)
        else:
            # old branch scored on out-of-task (new) data -> negative statistics
            if index in self.calib_neg_stats:
                m = self.calib_neg_momentum
                old_mu, old_sigma = self.calib_neg_stats[index]
                self.calib_neg_stats[index] = (m * old_mu + (1 - m) * mu, m * old_sigma + (1 - m) * sigma)
            else:
                self.calib_neg_stats[index] = (mu, sigma)
        logging.info(
            "[Branch Calib] task=%s branch=%s kind=%s mu=%.4f sigma=%.4f",
            self._cur_task,
            index,
            "pos" if index == self._cur_task else "neg",
            mu,
            sigma,
        )

        if self.dump_eval:
            np.savez_compressed(
                os.path.join(self._analysis_dir(), "train_emb_task{}_branch{}.npz".format(self._cur_task, index)),
                embeddings=embedding_list.numpy().astype(np.float16),
                labels=label_list.numpy().astype(np.int32),
            )

    def _branch_calib_params(self, index):
        if self.calib_scheme in ("pos_zscore", "pos_shift"):
            return self.calib_pos_stats.get(index, (None, None))
        if self.calib_scheme == "neg_zscore":
            # newest branch has no negatives yet -> fall back to its pos stats
            if index in self.calib_neg_stats:
                return self.calib_neg_stats[index]
            return self.calib_pos_stats.get(index, (None, None))
        return (None, None)

    def _calibrate_logits(self, outputs):
        """Per-branch affine calibration of the concatenated diagonal logits.

        Within a branch the map is monotone (ranking unchanged); only the
        cross-branch comparison — i.e. the implicit task-id decision — changes."""
        if not self.calib_enable or self.calib_scheme in (None, "none"):
            return outputs
        calibrated = outputs.clone()
        for i in range(self._cur_task + 1):
            start_cls, end_cls = self.get_cls_range(i)
            mu, sigma = self._branch_calib_params(i)
            if mu is None:
                continue
            if self.calib_scheme == "pos_shift":
                calibrated[:, start_cls:end_cls] = outputs[:, start_cls:end_cls] - mu
            else:
                denom = max(sigma if sigma is not None else 0.0, self.calib_eps)
                calibrated[:, start_cls:end_cls] = (outputs[:, start_cls:end_cls] - mu) / denom
        return calibrated

    def get_A_B_Ahat(self, task_id):
        if self.use_init_ptm:
            start_dim = (task_id + 1) * self._network.out_dim
            end_dim = start_dim + self._network.out_dim
        else:
            start_dim = task_id * self._network.out_dim
            end_dim = start_dim + self._network.out_dim

        start_cls, end_cls = self.get_cls_range(task_id)

        # W(Ti)  i is the i-th task index, T is the cur task index, W is a T*T matrix
        A = self._network.fc.weight.data[self._known_classes:, start_dim : end_dim]
        #A = self._network.fc.weight.data[0:, start_dim : end_dim]
        # W(TT)
        B = self._network.fc.weight.data[self._known_classes:, -self._network.out_dim:]
        #B = self._network.fc.weight.data[0:, -self._network.out_dim:]
        # W(ii)
        A_hat = self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim]

        return A.cpu(), B.cpu(), A_hat.cpu()

    def solve_similarity(self):
        for task_id in range(self._cur_task):
            # print('Solve_similarity adapter:{}'.format(task_id))
            start_cls, end_cls = self.get_cls_range(task_id=task_id)

            A, B, A_hat = self.get_A_B_Ahat(task_id=task_id)

            # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
            similarity = torch.zeros(len(A_hat), len(A))
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

            # softmax the similarity, it will be failed if not use it
            similarity = F.softmax(similarity, dim=1)

            # weight the combination of B(new_cls2)
            B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    B_hat[i] += similarity[i][j] * B[j]

            # B_hat(old_cls2)
            self._network.fc.weight.data[start_cls : end_cls, -self._network.out_dim:] = B_hat.to(self._device)

    def solve_sim_reset(self):
        for task_id in range(self._cur_task):
            if self.moni_adam and task_id > self.adapter_num - 2:
                break

            if self.use_init_ptm:
                range_dim = range(task_id + 2, self._cur_task + 2)
            else:
                range_dim = range(task_id + 1, self._cur_task + 1)
            for dim_id in range_dim:
                if self.moni_adam and dim_id > self.adapter_num:
                    break
                # print('Solve_similarity adapter:{}, {}'.format(task_id, dim_id))
                start_cls, end_cls = self.get_cls_range(task_id=task_id)

                start_dim = dim_id * self._network.out_dim
                end_dim = (dim_id + 1) * self._network.out_dim

                # Use the element above the diagonal to calculate
                if self.use_init_ptm:
                    start_cls_old = self.init_cls + (dim_id - 2) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = (task_id + 1) * self._network.out_dim
                    end_dim_old = (task_id + 2) * self._network.out_dim
                else:
                    start_cls_old = self.init_cls + (dim_id - 1) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = task_id * self._network.out_dim
                    end_dim_old = (task_id + 1) * self._network.out_dim

                A = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_old:end_dim_old].cpu()
                B = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim:end_dim].cpu()
                A_hat = self._network.fc.weight.data[start_cls:end_cls, start_dim_old:end_dim_old].cpu()

                # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
                similarity = torch.zeros(len(A_hat), len(A))
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

                # softmax the similarity, it will be failed if not use it
                similarity = F.softmax(similarity, dim=1) # dim=1, not dim=0

                # weight the combination of B(new_cls2)
                B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        B_hat[i] += similarity[i][j] * B[j]

                # B_hat(old_cls2)
                self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim] = B_hat.to(self._device)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        if hasattr(self._network.backbone, "log_sd_lora_rr_state"):
            self._network.backbone.log_sd_lora_rr_state()

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # --- EWC on shared adapter: estimate this task's Fisher (CE only),
        # accumulate it, and save the current A_s as the reference for the
        # next task. Done here (before add_fc) because the Fisher forward uses
        # proxy_fc, which add_fc() deletes. No optimizer.step() happens here.
        if self.ewc.enable:
            backbone = self._network.backbone
            fisher_cur, sample_count, _ = self.ewc.estimate_fisher(self)
            self.ewc.update_fisher(fisher_cur)
            self.ewc.log_fisher(self._cur_task, fisher_cur, sample_count)
            self.ewc.save_reference(backbone)
            logging.info("[EWC Ref]")
            logging.info("Saved shared adapter reference A_s after task {}".format(self._cur_task))

        self._network.add_fc()
        self.replace_fc(self.train_loader_for_protonet)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            # for base 0 setting, the later_lr and later_epochs are not used
            # for base N setting, the later_lr and later_epochs are used
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]

            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)


    def _optimizer_param_groups(self):
        # B8: weight decay on the scale/routing parameters (direction_scale,
        # block_weight, cosine-head sigma) systematically shrinks routing
        # weights toward 0 every step; config no_decay_on_scales exempts them.
        trainable = [
            (n, p) for n, p in self._network.named_parameters() if p.requires_grad
        ]
        if not self.args.get("no_decay_on_scales", False):
            return [{"params": [p for _, p in trainable], "weight_decay": self.weight_decay}]
        scale_keys = ("direction_scale", "block_weight", "sigma")
        decay, no_decay = [], []
        for name, param in trainable:
            (no_decay if any(k in name for k in scale_keys) else decay).append(param)
        return [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def get_optimizer(self, lr):
        param_groups = self._optimizer_param_groups()
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                lr=lr,
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                param_groups,
                lr=lr,
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                lr=lr,
            )

        return optimizer

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self.moni_adam:
            if self._cur_task > self.adapter_num - 1:
                return

        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']

        backbone = self._network.module.backbone if isinstance(self._network, nn.DataParallel) else self._network.backbone
        if self.ewc.enable:
            self.ewc.log_param_check(backbone)

        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            cls_losses = 0.0
            kd_losses = 0.0
            orth_losses = 0.0
            dir_orth_losses = 0.0
            ewc_losses = 0.0
            correct, total = 0, 0

            if not self._network.backbone.msa_adapt:

                for name, param in self._network.backbone.cur_adapter[0].named_parameters():
                    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
            else:
                for name, param in self._network.backbone.cur_adapter[0][1].named_parameters():
                    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
                for name, param in self._network.backbone.cur_adapter[-1][1].named_parameters():
                    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")


            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                aux_targets = targets.clone()

                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )
                output = self._network(inputs, test=False)

                logits = output["logits"]

                loss_cls = F.cross_entropy(logits, aux_targets)
                loss = loss_cls
                loss_kd = logits.new_zeros(())

                if self._cur_task > 0:
                    kd_ratio = 5.
                    Temperature = 2

                    out_new, out_teacher = self._network.forward_kd(inputs, self._cur_task)
                    out_new_logits = out_new["logits"]
                    out_teacher_logits = out_teacher["logits"]
                    loss_kd = kd_ratio * _KD_loss(out_new_logits, out_teacher_logits, T=Temperature)
                    loss = loss + loss_kd

                orth_loss_specific = logits.new_zeros(())
                if self._cur_task > 0 and self.use_orthogonal_constraint:
                    backbone = self._network.backbone
                    orth_loss_specific = compute_optional_orthogonality_loss(
                        getattr(backbone, "block_weight_list", []),
                        getattr(backbone, "block_weight", None),
                        self.use_orthogonal_constraint,
                    )
                    if self.use_orthogonal_constraint and self.orthogonal_lambda != 0.0:
                        loss = loss + self.orthogonal_lambda * orth_loss_specific

                # B5: direction-subspace orthogonality on task-specific adapters.
                dir_orth_loss = logits.new_zeros(())
                if self._cur_task > 0 and self.direction_orth_lambda > 0.0:
                    dir_orth = self._network.backbone.sd_lora_direction_orth_loss()
                    if dir_orth is not None:
                        dir_orth_loss = dir_orth
                        loss = loss + self.direction_orth_lambda * dir_orth_loss

                # EWC penalty on the shared (general_pos) adapters only. Returns
                # None on task 0 (no cumulative Fisher yet) -> treated as 0.
                ewc_loss = self.ewc.compute_ewc_loss(backbone)
                if ewc_loss is not None:
                    loss = loss + ewc_loss

                # Single backward over CE + KD + orth + EWC so every update on the
                # shared adapters is covered by the EWC penalty and only one
                # optimizer.step() happens per batch (no momentum cross-talk).
                # set_to_none=True avoids stale-momentum "ghost" steps on torch<2.0.
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient redistribution on the shared adapters: rescale each row
                # of lora_A.grad by the (mean-1 normalized) row norms of the
                # previous task's adapter, so rows that were important for old
                # tasks receive stronger corrective gradient. Applied to the
                # combined gradient between backward and step.
                if self._cur_task > 0:
                    for j in range(len(self._network.backbone.general_pos)):
                        pos = self._network.backbone.adapt_pos.index(self._network.backbone.general_pos[j])
                        for jj in range(len(self._network.backbone.msa)):
                            if self._network.backbone.msa[jj] == 1:
                                grad = self._network.backbone.cur_adapter[pos][jj].lora_A.weight.grad
                                if grad is None:
                                    continue
                                temp_weights = 1. * torch.norm(self._network.backbone.old_adapter_list[self._cur_task-1][pos][jj].lora_A.weight,dim=1)
                                temp_weights = 1. * len(temp_weights) * temp_weights / torch.sum(temp_weights)
                                self._network.backbone.cur_adapter[pos][jj].lora_A.weight.grad = temp_weights.unsqueeze(1) * grad

                optimizer.step()
                losses += loss.item()
                cls_losses += loss_cls.item()
                kd_losses += loss_kd.item()
                orth_losses += orth_loss_specific.item()
                dir_orth_losses += dir_orth_loss.item()
                if ewc_loss is not None:
                    ewc_losses += ewc_loss.item()
                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Cls_loss {:.3f}, KD_loss {:.3f}, Orth_loss {:.6f}, DirOrth_loss {:.6f}, EWC_loss {:.6f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    cls_losses / len(train_loader),
                    kd_losses / len(train_loader),
                    orth_losses / len(train_loader),
                    dir_orth_losses / len(train_loader),
                    ewc_losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)


    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _task_id_of(self, class_ids):
        return torch.clamp((class_ids - self.init_cls) // self.inc + 1, min=0)

    def _log_task_metrics(self, raw_logits, calibrated, scores, targets):
        true_task = self._task_id_of(targets)

        def task_id_acc(matrix):
            pred_task = self._task_id_of(matrix.argmax(dim=1))
            return (pred_task == true_task).float().mean().item() * 100

        # oracle: restrict the final scores to the true task's class block
        oracle_correct = 0
        for i in range(self._cur_task + 1):
            start_cls, end_cls = self.get_cls_range(i)
            mask = (targets >= start_cls) & (targets < end_cls)
            if mask.any():
                block_pred = start_cls + scores[mask][:, start_cls:end_cls].argmax(dim=1)
                oracle_correct += (block_pred == targets[mask]).sum().item()

        logging.info("Task correct: {}".format(task_id_acc(scores)))
        if self.calib_enable:
            logging.info("Task correct (raw, uncalibrated): {}".format(task_id_acc(raw_logits)))
        if scores is not calibrated:
            logging.info("Task correct (diag calibrated): {}".format(task_id_acc(calibrated)))
        logging.info("Task acc: {}".format(oracle_correct * 100.0 / len(targets)))

    def _eval_cnn(self, loader):
        self._network.eval()
        need_features = self.ridge_enable or self.dump_eval
        logits_batches, feature_batches, target_batches = [], [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                eval_out = self._network.forward(inputs, test=True)
            logits_batches.append(eval_out["logits"].cpu())
            if need_features:
                feature_batches.append(eval_out["features"].cpu())
            target_batches.append(targets)

        raw_logits = torch.cat(logits_batches, dim=0).float()
        targets_all = torch.cat(target_batches, dim=0)
        features_all = torch.cat(feature_batches, dim=0).float() if need_features else None

        # B1: per-branch affine calibration; identity when disabled
        calibrated = self._calibrate_logits(raw_logits)
        scores = calibrated

        # B3: ridge head in the branch-0 space, optionally ensembled with the
        # calibrated diagonal logits (both globally z-scored first).
        if self.ridge_enable:
            ridge_W = self._ridge_solve()
            if ridge_W is not None:
                x0 = F.normalize(features_all[:, : self._network.out_dim], p=2, dim=1)
                ridge_logits = x0 @ ridge_W
                if self.ridge_mode == "ridge_only":
                    scores = ridge_logits
                else:
                    scores = (
                        self.ridge_weight * self._zscore(ridge_logits)
                        + (1.0 - self.ridge_weight) * self._zscore(calibrated)
                    )

        y_pred = torch.topk(scores, k=self.topk, dim=1, largest=True, sorted=True)[1].numpy()
        y_true = targets_all.numpy()

        self._log_task_metrics(raw_logits, calibrated, scores, targets_all)

        if self.dump_eval:
            pos_items = sorted(self.calib_pos_stats.items())
            neg_items = sorted(self.calib_neg_stats.items())
            np.savez_compressed(
                os.path.join(self._analysis_dir(), "eval_task{}.npz".format(self._cur_task)),
                raw_logits=raw_logits.numpy().astype(np.float16),
                targets=targets_all.numpy().astype(np.int32),
                features=features_all.numpy().astype(np.float16),
                pos_stats_branches=np.array([k for k, _ in pos_items], dtype=np.int32),
                pos_stats=np.array([v for _, v in pos_items], dtype=np.float32),
                neg_stats_branches=np.array([k for k, _ in neg_items], dtype=np.int32),
                neg_stats=np.array([v for _, v in neg_items], dtype=np.float32),
                init_cls=np.int32(self.init_cls),
                increment=np.int32(self.inc),
                cur_task=np.int32(self._cur_task),
            )

        return y_pred, y_true  # [N, topk]
