"""Attribute resident GPU memory growth across 40 simulated tasks (B7 diagnosis).

Builds OurNet from a config json, then repeats the trainer's per-task growth
sequence (update_fc -> freeze -> add_adapter_to_list) without any training,
reporting per-component parameter+buffer bytes and torch.cuda.memory_allocated
after each task. Run:
    CUDA_VISIBLE_DEVICES=3 python tools/vram_attribution.py exps/inr_gfinal.json
"""
import json
import sys
import os
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.inc_net import OurNet


def component_bytes(module):
    """Sum param+buffer bytes grouped by the module's top-level attribute."""
    sizes = collections.Counter()
    for name, p in module.named_parameters():
        sizes[name.split(".")[0]] += p.numel() * p.element_size()
    for name, b in module.named_buffers():
        sizes[name.split(".")[0]] += b.numel() * b.element_size()
    return sizes


def cache_bytes(module):
    """Bytes held by SD-LoRA cached_old_direction_weights buffers, by owner."""
    total = collections.Counter()
    for name, m in module.named_modules():
        w = getattr(m, "cached_old_direction_weights", None)
        if isinstance(w, torch.Tensor) and w.numel() > 0:
            owner = name.split(".")[0]
            total[owner] += w.numel() * w.element_size()
    return total


def fmt(sizes, top=8):
    items = sorted(sizes.items(), key=lambda kv: -kv[1])[:top]
    return "  ".join("{}={:.1f}MB".format(k, v / 1024 / 1024) for k, v in items)


def train_step(net, x):
    net.train()
    out = net(x, test=False)
    loss = out["logits"].float().pow(2).mean()
    if net._cur_task > 0:
        out_new, out_teacher = net.forward_kd(x, net._cur_task)
        loss = loss + (out_new["logits"] - out_teacher["logits"]).pow(2).mean()
    loss.backward()
    for p in net.parameters():
        p.grad = None


def eval_sweep(net, x):
    net.eval()
    with torch.no_grad():
        net(x, test=True)


def mem_line():
    torch.cuda.synchronize()
    return "peak_alloc={:.0f}MB reserved={:.0f}MB max_reserved={:.0f}MB".format(
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024,
        torch.cuda.max_memory_reserved() / 1024 / 1024,
    )


def main():
    with open(sys.argv[1]) as f:
        args = json.load(f)
    nb_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    args["device"] = [torch.device("cuda:0")]

    net = OurNet(args, True)
    net.to(args["device"][0])
    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated()
    print("after build: allocated={:.0f}MB".format(base_alloc / 1024 / 1024))

    init_cls, inc = args["init_cls"], args["increment"]
    batch_size = int(args.get("batch_size", 32))
    x = torch.randn(batch_size, 3, 224, 224, device=args["device"][0])
    total = 0
    for t in range(nb_tasks):
        total += init_cls if t == 0 else inc
        net.update_fc(total)
        net.to(args["device"][0])
        torch.cuda.reset_peak_memory_stats()
        # Real per-task order: train -> freeze+snapshot -> eval over all branches.
        train_step(net, x)
        net.freeze()
        net.backbone.add_adapter_to_list()
        net.to(args["device"][0])
        eval_sweep(net, x)
        if os.environ.get("EMPTY_CACHE_PER_TASK") == "1":
            torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1024 / 1024
        print("task {:2d}: resident={:.0f}MB {}".format(t + 1, alloc, mem_line()))
        if (t + 1) % 10 == 0 or t == nb_tasks - 1:
            bb = component_bytes(net.backbone)
            caches = cache_bytes(net.backbone)
            print("         backbone: {}".format(fmt(bb)))
            print("         direction caches total={:.1f}MB".format(
                sum(caches.values()) / 1024 / 1024))


if __name__ == "__main__":
    main()
