import copy

import torch
from torch import nn

from backbone.linears import CosineLinearFeature


def parse_cllora_adapter_options(args, ffn_num):
    """Resolve the SD-LoRA / task-specific-adapter options with method defaults.

    Defaults correspond to the released method (SD-LoRA-RR task-conditioned
    directions, rank decay 0.9); any key can be overridden from the config's
    "sd_lora" / "task_specific_adapter" blocks.
    """
    sd_cfg = args.get("sd_lora", {}) or {}
    tsa_cfg = args.get("task_specific_adapter", {}) or {}

    return {
        "sd_lora_enable": bool(sd_cfg.get("enable", True)),
        "sd_lora_variant": sd_cfg.get("variant", "rr"),
        "sd_lora_r0": int(sd_cfg.get("r0", ffn_num)),
        "sd_lora_r_min": int(sd_cfg.get("r_min", 1)),
        "sd_lora_rank_decay": float(sd_cfg.get("rank_decay", 0.9)),
        "normalize_direction": bool(sd_cfg.get("normalize_direction", True)),
        "alpha_mode": sd_cfg.get("alpha_mode", "task_conditioned"),
        "train_old_alpha": bool(sd_cfg.get("train_old_alpha", False)),
        "alpha_init": float(sd_cfg.get("alpha_init", args.get("direction_scale_init", 0.05))),
        "direction_norm_eps": float(sd_cfg.get("direction_norm_eps", args.get("direction_norm_eps", 1e-6))),
        "cache_old_directions": bool(sd_cfg.get("cache_old_directions", True)),
        "cache_device": sd_cfg.get("cache_device", "cuda"),
        "combine_directions_before_linear": bool(sd_cfg.get("combine_directions_before_linear", True)),
        "use_block_weight": bool(tsa_cfg.get("use_block_weight", args.get("use_block_weight", True))),
        "use_orth_loss": bool(tsa_cfg.get("use_orth_loss", args.get("use_orthogonal_constraint", False))),
    }


def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    if name not in ("vit_base_patch16_224_cllora", "vit_base_patch16_224_in21k_cllora"):
        raise NotImplementedError("Unknown backbone type {}".format(name))

    from backbone import vit_cllora
    from easydict import EasyDict

    ffn_num = args.get("ffn_num", 10)  # rank of the shared (general_pos) LoRA adapters
    adapter_options = parse_cllora_adapter_options(args, ffn_num)

    tuning_config = EasyDict(
        use_distillation=args.get("use_distillation", True),
        use_block_weight=adapter_options["use_block_weight"],
        use_orthogonal_constraint=adapter_options["use_orth_loss"],
        orthogonal_lambda=args.get("orthogonal_lambda", 0.0),
        # adapter placement: msa = [attn, ffn-serial, ffn-parallel] channels,
        # shared adapters on the first 6 blocks, task-specific on the last 6
        msa_adapt=args.get("msa_adapt", True),
        msa=args.get("msa", [1, 0, 1]),
        specific_pos=args.get("specific_pos", args.get("specfic_pos", [6, 7, 8, 9, 10, 11])),
        general_pos=args.get("general_pos", [0, 1, 2, 3, 4, 5]),
        # SD-LoRA-RR task-specific adapter
        sd_lora_enable=adapter_options["sd_lora_enable"],
        sd_lora_variant=adapter_options["sd_lora_variant"],
        sd_lora_r0=adapter_options["sd_lora_r0"],
        sd_lora_r_min=adapter_options["sd_lora_r_min"],
        sd_lora_rank_decay=adapter_options["sd_lora_rank_decay"],
        normalize_direction=adapter_options["normalize_direction"],
        alpha_mode=adapter_options["alpha_mode"],
        train_old_alpha=adapter_options["train_old_alpha"],
        direction_scale_init=adapter_options["alpha_init"],
        direction_norm_eps=adapter_options["direction_norm_eps"],
        cache_old_directions=adapter_options["cache_old_directions"],
        cache_device=adapter_options["cache_device"],
        combine_directions_before_linear=adapter_options["combine_directions_before_linear"],
        block_weight_norm_eps=args.get("block_weight_norm_eps", 1e-6),
        block_weight_normalization=args.get("block_weight_normalization", "mean_l1"),
        specific_lora_init_scale=args.get("specific_lora_init_scale", 1e-3),
        nb_tasks=args.get("nb_tasks", 1),
        # B7a: evaluate every branch with the CURRENT shared adapters (original
        # CL-LoRA semantics, ~35% faster eval) instead of per-task snapshots.
        eval_shared_current=args.get("eval_shared_current", False),
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=ffn_num,
        d_model=768,
        vpt_on=False,
        vpt_num=0,
        _device=args["device"][0],
    )
    if name == "vit_base_patch16_224_cllora":
        model = vit_cllora.vit_base_patch16_224_cllora(
            num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
    else:
        model = vit_cllora.vit_base_patch16_224_in21k_cllora(
            num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
    model.out_dim = 768
    return model.eval()


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        self.fc = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class OurNet(BaseNet):
    """CL-LoRA network: one cosine-prototype block per task branch.

    fc is a CosineLinearFeature over the concatenation of all branch features;
    forward_diagonal scores each branch's classes only against that branch's
    own feature block (the pure-diagonal routing head).
    """

    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim = self.backbone.out_dim
        self.fc = None
        self.fc_list = nn.ModuleList()
        self.init_proto = None

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    @property
    def feature_dim(self):
        return self.out_dim * (self._cur_task + 1)

    def update_fc(self, nb_classes):
        self._cur_task += 1

        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)
        init_proto = self.generate_fc(self.out_dim, nb_classes).to(self._device)

        if self.init_proto is not None:
            old_nb_classes = self.init_proto.out_features
            weight = copy.deepcopy(self.init_proto.weight.data)
            init_proto.weight.data[:old_nb_classes, :] = nn.Parameter(weight)
        del self.init_proto
        self.init_proto = init_proto

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()

        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[:old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc
        self.fc.requires_grad_(False)

    def add_fc(self):
        self.fc_list.append(self.proxy_fc.requires_grad_(False))
        del self.proxy_fc

    def generate_fc(self, in_dim, out_dim):
        return CosineLinearFeature(in_dim, out_dim)

    def extract_vector(self, x):
        return self.backbone(x)

    def forward_kd(self, x, t_idx):
        x_new, x_teacher = self.backbone.forward_general_cls(x, t_idx)
        return self.proxy_fc(x_new), self.proxy_fc(x_teacher)

    def forward(self, x, test=False):
        if not test:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
            out.update({"features": x})
            return out
        x_input = self.backbone.forward(x, True)
        out = self.fc.forward_diagonal(
            x_input, cur_task=self._cur_task, init_cls=self.init_cls, inc=self.inc)
        out.update({"features": x_input})
        return out
