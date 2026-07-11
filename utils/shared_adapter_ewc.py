import logging
import torch
import torch.nn.functional as F


class SharedAdapterEWC:
    """EWC regularization applied ONLY to CL-LoRA's shared (general_pos) adapters.

    The shared adapter update is Delta_W_s = A_s @ B_s, where:
      * A_s = adapter.lora_A.weight  -> trainable, persists / keeps updating across tasks
      * B_s = adapter.lora_B.weight  -> fixed orthogonal, frozen (requires_grad=False)

    We constrain the *drift* of the full update relative to the reference saved at the
    end of the previous task:
        D_t = (A_s - A_s_ref) @ B_s
        L_ewc = (lambda/2) * sum_j  F_cum_j * vec(D_t)_j^2

    Fisher is a diagonal estimate in the FULL Delta_W_s = A_s @ B_s space (shape
    [n_embd, n_embd]) obtained via retain_grad on delta_w during a CE-only forward.

    This class never touches task-specific adapters.
    """

    def __init__(self, args):
        ewc_cfg = args.get("ewc", {}) or {}
        self.enable = bool(ewc_cfg.get("enable", True))
        self.lam = float(ewc_cfg.get("lambda", 10.0))
        self.gamma = float(ewc_cfg.get("gamma", 1.0))
        self.fisher_sample_size = ewc_cfg.get("fisher_sample_size", None)
        self.fisher_batch_size = ewc_cfg.get("fisher_batch_size", None)
        self.fisher_on_shared_adapter_only = bool(ewc_cfg.get("fisher_on_shared_adapter_only", True))
        self.fisher_use_ce_only = bool(ewc_cfg.get("fisher_use_ce_only", True))
        self.store_full_delta_fisher = bool(ewc_cfg.get("store_full_delta_fisher", True))
        self.normalize_fisher = bool(ewc_cfg.get("normalize_fisher", False))

        self.device = args["device"][0]

        # cumulative Fisher and reference A_s, keyed by (block_id, msa_index)
        self.fisher_cum = {}
        self.A_ref = {}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _iter_shared_adapters(self, backbone):
        """Yield ((block_id, msa_index), adapter) for every real shared adapter.

        Shared adapters live at general_pos blocks; only msa channels with msa[j]==1
        are real Adapter_lora modules (others are nn.Identity)."""
        for g in backbone.general_pos:
            pos = backbone.adapt_pos.index(g)
            for j in range(len(backbone.msa)):
                if backbone.msa[j] == 1:
                    adapter = backbone.cur_adapter[pos][j]
                    # only Adapter_lora has lora_A / lora_B
                    if hasattr(adapter, "lora_A") and hasattr(adapter, "lora_B"):
                        yield (g, j), adapter

    # ------------------------------------------------------------------ #
    # EWC loss (added to the training total loss)
    # ------------------------------------------------------------------ #
    def compute_ewc_loss(self, backbone):
        """Return the EWC penalty as a differentiable scalar tensor, or None.

        Returns None when EWC is disabled or no cumulative Fisher exists yet
        (i.e. during the first task), in which case the caller treats it as 0."""
        if not self.enable or len(self.fisher_cum) == 0:
            return None

        total = None
        for key, adapter in self._iter_shared_adapters(backbone):
            if key not in self.fisher_cum or key not in self.A_ref:
                continue
            A = adapter.lora_A.weight                 # [n_embd, down_size], trainable
            B = adapter.lora_B.weight                 # [down_size, n_embd], frozen
            A_ref = self.A_ref[key]                    # detached reference
            D = (A - A_ref) @ B                        # [n_embd, n_embd] = drift of Delta_W_s
            F_cum = self.fisher_cum[key]               # [n_embd, n_embd]
            term = (F_cum * D.pow(2)).sum()
            total = term if total is None else total + term

        if total is None:
            return None
        return 0.5 * self.lam * total

    # ------------------------------------------------------------------ #
    # Fisher estimation (CE/NLL only, no optimizer.step)
    # ------------------------------------------------------------------ #
    @torch.enable_grad()
    def estimate_fisher(self, learner):
        """Estimate the diagonal Fisher of Delta_W_s = A_s @ B_s on the current task.

        Uses ONLY the classification CE loss (no KD/orth/EWC/alpha). Does NOT call
        optimizer.step(). Returns (fisher_cur dict, sample_count, n_batches)."""
        network = learner._network
        backbone = network.backbone
        device = learner._device
        loader = learner.train_loader
        # B6: the gradient of the batch-mean CE has near-zero-mean per-sample
        # noise that cancels, so grad² of a size-B batch underestimates the
        # Fisher by ~1/B (this is why lambda had to be ~180). fisher_batch_size
        # re-batches the estimation pass; 1 gives the true per-sample Fisher,
        # and an existing lambda should be rescaled by ~old_bs/new_bs.
        if self.fisher_batch_size:
            from torch.utils.data import DataLoader
            loader = DataLoader(
                learner.train_dataset,
                batch_size=int(self.fisher_batch_size),
                shuffle=False,
                num_workers=loader.num_workers,
            )
        known_classes = learner._known_classes

        adapters = dict(self._iter_shared_adapters(backbone))

        # enable fisher mode on shared adapters so forward exposes delta_w with grad
        for adapter in adapters.values():
            adapter.ewc_fisher_mode = True
            adapter.ewc_delta_w = None

        was_training = network.training
        network.eval()  # disable dropout; params still hold grad

        fisher_cur = {}
        n_batches = 0
        sample_count = 0

        for _, (_, inputs, targets) in enumerate(loader):
            if self.fisher_sample_size is not None and sample_count >= self.fisher_sample_size:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            # same label shift as training (current-task-relative labels)
            aux_targets = targets.clone()
            aux_targets = torch.where(
                aux_targets - known_classes >= 0,
                aux_targets - known_classes,
                -1,
            )

            network.zero_grad(set_to_none=True)
            output = network(inputs, test=False)
            logits = output["logits"]
            loss = F.cross_entropy(logits, aux_targets)  # CE/NLL only
            loss.backward()

            for key, adapter in adapters.items():
                dw = adapter.ewc_delta_w
                if dw is None or dw.grad is None:
                    continue
                sq = dw.grad.detach().pow(2)
                if key not in fisher_cur:
                    fisher_cur[key] = torch.zeros_like(sq)
                fisher_cur[key] += sq
                adapter.ewc_delta_w = None  # release graph reference

            n_batches += 1
            sample_count += inputs.size(0)

        # average over batches
        if n_batches > 0:
            for key in fisher_cur:
                fisher_cur[key] /= n_batches

        if self.normalize_fisher:
            for key in fisher_cur:
                m = fisher_cur[key].max()
                if m > 0:
                    fisher_cur[key] = fisher_cur[key] / m

        # restore normal mode and clean up
        for adapter in adapters.values():
            adapter.ewc_fisher_mode = False
            adapter.ewc_delta_w = None
        network.zero_grad(set_to_none=True)
        if was_training:
            network.train()

        return fisher_cur, sample_count, n_batches

    def update_fisher(self, fisher_cur):
        """Accumulate Fisher:  F_cum = gamma * F_cum + F_t."""
        for key, f in fisher_cur.items():
            if key in self.fisher_cum:
                self.fisher_cum[key] = self.gamma * self.fisher_cum[key] + f
            else:
                self.fisher_cum[key] = f.clone()

    def save_reference(self, backbone):
        """Save A_s_ref <- stopgrad(A_s) as a detached clone (no graph)."""
        for key, adapter in self._iter_shared_adapters(backbone):
            self.A_ref[key] = adapter.lora_A.weight.detach().clone()

    # ------------------------------------------------------------------ #
    # logging
    # ------------------------------------------------------------------ #
    def log_config(self):
        logging.info("EWC enabled: {}".format(self.enable))
        logging.info("EWC lambda: {}".format(self.lam))
        logging.info("EWC gamma: {}".format(self.gamma))

    def log_param_check(self, backbone):
        shared_A_trainable = None
        shared_B_trainable = None
        for _, adapter in self._iter_shared_adapters(backbone):
            shared_A_trainable = bool(adapter.lora_A.weight.requires_grad)
            shared_B_trainable = bool(adapter.lora_B.weight.requires_grad)
            break
        logging.info("[EWC Param Check]")
        logging.info("shared_A_trainable: {}".format(shared_A_trainable))
        logging.info("shared_B_trainable: {}".format(shared_B_trainable))
        # EWC only ever touches shared adapters (general_pos); task-specific never.
        logging.info("ewc_on_task_specific: {}".format(False))

    def log_fisher(self, task_id, fisher_cur, sample_count):
        num = len(fisher_cur)
        logging.info("[EWC Fisher]")
        logging.info("task_id: {}".format(task_id))
        logging.info("num_shared_adapters: {}".format(num))
        logging.info("fisher_sample_count: {}".format(sample_count))
        if num == 0:
            logging.info("fisher_mean: 0")
            logging.info("fisher_max: 0")
            logging.info("fisher_min: 0")
            logging.info("fisher_nonzero_ratio: 0")
            return
        flat = torch.cat([f.flatten() for f in fisher_cur.values()])
        nonzero_ratio = (flat > 0).float().mean().item()
        logging.info("fisher_mean: {:.6e}".format(flat.mean().item()))
        logging.info("fisher_max: {:.6e}".format(flat.max().item()))
        logging.info("fisher_min: {:.6e}".format(flat.min().item()))
        logging.info("fisher_nonzero_ratio: {:.4f}".format(nonzero_ratio))
