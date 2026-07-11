# CL-LoRA + SD-LoRA-RR + EWC: Rehearsal-Free Class-Incremental Learning with a Dual-Head Ensemble

This repository extends [CL-LoRA (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.html) with:

- **SD-LoRA-RR task-specific adapters** — task-conditioned direction scaling with rank decay on the task-specific (last 6) blocks.
- **EWC on the shared adapters** — a diagonal-Fisher penalty on the drift of the shared (first 6) blocks' LoRA update, with Fisher decay (`gamma`) for long task sequences.
- **A rehearsal-free dual-head inference ensemble** — the key finding of this work: the accuracy bottleneck of the pure-diagonal cosine head is *cross-task score calibration* (task-id routing), not representation forgetting. We fix it at inference only:
  1. per-branch **pos_shift calibration** of the diagonal logits, and
  2. an **online ridge head** in the frozen branch-0 feature space (only a Gram matrix `G` and target matrix `C` are accumulated — no samples stored), z-score-ensembled with the calibrated diagonal logits at weight `w = 0.15`.

  One global configuration (`pos_shift + ridge, w = 0.15`) is non-negative vs. the training-identical baseline on all four benchmarks.

## Environment

- `python==3.9`, `torch==2.0.1`, `torchvision==0.15.2`, `timm==0.6.12`
- `numpy`, `scikit-learn`, `easydict`, `tqdm`

```bash
conda create -n cl_lora python=3.9
conda activate cl_lora
pip install torch==2.0.1 torchvision==0.15.2 timm==0.6.12 numpy scikit-learn easydict tqdm
```

The ViT-B/16 (IN21k) backbone weights are downloaded automatically by `timm`.

## Data preparation

Datasets are **not** included in this repository. Place them under `./data/`:

| Dataset | Layout |
|---|---|
| CIFAR-100 | auto-downloaded to `./data/` on first run |
| ImageNet-R | `./data/imagenet-r/{train,test}/<class>/*.jpg` |
| ImageNet-A | `./data/imagenet-a/{train,test}/<class>/*.jpg` |
| VTAB (5 tasks) | `./data/vtab/{train,test}/<class>/*.jpg` |

ImageNet-R/A and VTAB splits follow the protocol of [ADAM / PILOT](https://github.com/sun-hailong/LAMDA-PILOT); see `utils/data.py` for the exact paths.

## Training

```bash
python main.py exps/cifar.json    # CIFAR-100,   5 cls x 20 tasks, 30 epochs
python main.py exps/ina.json      # ImageNet-A, 20 cls x 10 tasks, 25 epochs
python main.py exps/inr.json      # ImageNet-R,  5 cls x 40 tasks, 25 epochs
python main.py exps/vtab.json     # VTAB,       10 cls x  5 tasks, 45 epochs
```

Logs (accuracy curve, forgetting matrix, task-id routing accuracy) are written to `logs/cllora/<dataset>/...`.

### Adapter placement (fixed defaults)

- `msa = [1, 0, 1]`: adapt Q and V in multi-head self-attention
- `general_pos = [0..5]`: task-shared LoRA in the first 6 ViT blocks
- `specific_pos = [6..11]`: task-specific LoRA in the last 6 ViT blocks

## Expected results (seed 1993, single run)

`Base` = same training, inference with the raw diagonal head only
(`"branch_calibration": {"enable": false}, "ridge_head": {"enable": false}`).
`Ours` = the released configs (calibration + ridge ensemble, `w = 0.15`).

| Benchmark | Base avg / final / forgetting | Ours avg / final / forgetting | Peak VRAM |
|---|---|---|---|
| CIFAR-100 (5×20) | 92.39 / 87.49 / 6.09 | **93.98 / 90.08 / 4.04** ¹ | 11.5 GB |
| ImageNet-A (20×10) | 69.50 / 59.38 / 9.89 | **72.44 / 62.01 / 10.18** | 6.1 GB |
| ImageNet-R (5×40) | 80.69 / 72.72 / 6.62 | **83.53 / 76.47 / 7.01** | ≤9.4 GB ² |
| VTAB (10×5) | 94.71 / 93.51 / 0.95 | **95.21 / 94.64 / 0.89** | 5.8 GB |

¹ collected at `ensemble_weight = 0.2`; the offline sweep predicts 93.94 at `w = 0.15` (run-to-run noise is ±0.35).
² after the per-task `torch.cuda.empty_cache()` fix; training is not bit-deterministic across GPUs/driver versions, so expect small deviations.

Note for ImageNet-R (40 tasks): `ewc.gamma = 0.9` is required — with `gamma = 1.0` the accumulated Fisher grows unboundedly and training diverges around task 37.

## Ablations

`exps/ablations/` keeps the configs used for the paper's ablations (baseline
inference, ridge-only, no-EWC, `eval_shared_current` fast path, B6/B8 variants).
Every knob can also be toggled directly in a release config, e.g.:

```jsonc
"branch_calibration": {"enable": false},          // no calibration
"ridge_head": {"enable": false},                  // no ridge ensemble
"ridge_head": {"mode": "ridge_only"},             // ridge as the only head
"ewc": {"enable": false},                         // no EWC
"eval_shared_current": true                       // ~35% faster eval (B7a)
```

Analysis tools (`tools/`): `sweep_ensemble.py` (offline scheme×weight sweep on
`dump_eval` dumps), `compare_calibration.py` (calibration schemes),
`compare_heads.py` (ridge vs RanPAC vs FeCAM), `eval_b2_reweight.py`
(EASE-style completion ablation), `vram_attribution.py`, `exp_status.sh`.

## Citation

If you build on the CL-LoRA backbone mechanism, please cite:

```bibtex
@article{He_2025_CVPR,
    author    = {He, Jiangpeng and Duan, Zhihao and Zhu, Fengqing},
    title     = {CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning},
    journal = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {30534-30544}
}
```

## Acknowledgments

Built on [CL-LoRA](https://openaccess.thecvf.com/content/CVPR2025/html/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.html)
and the [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT) pre-trained CIL toolbox.

## License

MIT — see [LICENSE](LICENSE).
