"""Minimal structural smoke test for the SD-LoRA-RR task-specific adapter.

Runs a tiny ViT (no dataset / no pretrained weights) through 3 incremental
tasks for each ablation (A/B/C/D) and prints the quantities requested in the
spec: per-task rank, #new-direction params, old-direction frozen, alpha shape /
requires_grad, block-weight enabled, orth-loss enabled, and forward output dim.

Run:  conda run -n cl_lora python tests/smoke_sd_lora_rr.py
"""
from types import SimpleNamespace

import torch

from backbone.vit_cllora import SDLoRAAdapter, VisionTransformer


def make_config(**overrides):
    config = {
        "use_distillation": True,
        "use_block_weight": True,
        "use_orthogonal_constraint": False,
        "orthogonal_lambda": 0.0,
        "msa_adapt": True,
        "msa": [1, 0, 1],
        "specfic_pos": [2, 3],
        "general_pos": [0, 1],
        "ffn_adapt": True,
        "ffn_option": "parallel",
        "ffn_adapter_layernorm_option": "none",
        "ffn_adapter_init_option": "lora",
        "ffn_adapter_scalar": "0.1",
        "ffn_num": 8,
        "d_model": 8,
        "vpt_on": False,
        "vpt_num": 0,
        "_device": "cpu",
        "direction_scale_init": 0.05,
        "direction_norm_eps": 1e-6,
        "block_weight_normalization": "mean_l1",
        "specific_lora_init_scale": 1e-3,
        "specific_lora_rank_schedule": None,
        "nb_tasks": 3,
        # SD-LoRA-RR defaults (experiment D)
        "sd_lora_enable": True,
        "sd_lora_variant": "rr",
        "sd_lora_r0": 8,
        "sd_lora_r_min": 2,
        "sd_lora_rank_decay": 0.8,
        "normalize_direction": True,
        "alpha_mode": "task_conditioned",
        "train_old_alpha": False,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def build(**overrides):
    torch.manual_seed(7)
    return VisionTransformer(
        img_size=8, patch_size=4, in_chans=3, num_classes=0,
        embed_dim=8, depth=4, num_heads=2, mlp_ratio=2, qkv_bias=True,
        tuning_config=make_config(**overrides),
    )


def first_sd_adapter(model):
    pos = model.adapt_pos.index(model.specfic_pos[0])
    return model.cur_adapter[pos][0]


def inspect(model, task_id):
    adapter = first_sd_adapter(model)
    is_sd = getattr(adapter, "is_sd_lora_adapter", False)
    ranks = adapter.direction_ranks() if is_sd else []
    new_dir_params = sum(p.numel() for p in adapter.directions[-1].parameters()) if is_sd else 0
    old_frozen = all(
        not any(p.requires_grad for p in d.parameters())
        for d in adapter.directions[:-1]
    ) if is_sd else True
    cur_grad = (
        any(p.requires_grad for p in adapter.directions[-1].parameters())
        if is_sd else any(p.requires_grad for p in adapter.parameters())
    )
    print(
        f"  task {task_id}: rank={model.current_specific_rank} "
        f"direction_ranks={ranks} #new_dir_params={new_dir_params} "
        f"old_dirs_frozen={old_frozen} cur_dir_trainable={cur_grad} "
        f"alpha_shape={tuple(model.direction_scale.shape)} "
        f"alpha_req_grad={bool(model.direction_scale.requires_grad)} "
        f"old_alpha_req_grad={[bool(p.requires_grad) for p in model.direction_scale_list]} "
        f"block_weight={model.use_block_weight} "
        f"normalize={model.normalize_direction}"
    )


def run(name, **overrides):
    print(f"\n=== {name} ({overrides}) ===")
    model = build(**overrides)
    # freeze base like the pretrained loader would
    for n, p in model.named_parameters():
        if "cur_adapter" not in n and "block_weight" not in n and "direction_scale" not in n:
            p.requires_grad = False
    x = torch.randn(2, 3, 8, 8)
    for t in range(3):
        inspect(model, t)
        # 1 "epoch": one train forward+backward to confirm grad flow
        out = model.forward_train(x)
        out.pow(2).mean().backward()
        if t < 2:
            model.add_adapter_to_list()
    # inference forward dim check
    with torch.no_grad():
        feat = model.forward(x, test=True)
    expected = (len(model.adapter_list) + 1) * model.embed_dim  # snapshots + current branch
    print(f"  forward(test) output shape={tuple(feat.shape)} expected_width={expected} "
          f"ok={feat.shape[1] == expected and bool(torch.isfinite(feat).all())}")


if __name__ == "__main__":
    run("Experiment A: baseline (enable=false)",
        sd_lora_enable=False, sd_lora_variant="fixed", normalize_direction=False,
        alpha_mode="global")
    run("Experiment B: SD-LoRA fixed-rank",
        sd_lora_variant="fixed", alpha_mode="global")
    run("Experiment C: SD-LoRA-RR global alpha",
        sd_lora_variant="rr", alpha_mode="global")
    run("Experiment D: SD-LoRA-RR task_conditioned alpha (default)",
        sd_lora_variant="rr", alpha_mode="task_conditioned", train_old_alpha=False)
    print("\nAll smoke runs finished.")
