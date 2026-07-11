import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model
import torch.nn.functional as F
import numpy as np
import logging
import os
from collections import OrderedDict
import torch
import copy
import random
import re



class Adapter_lora(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.random_orth = True

        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.lora_A = nn.Linear(self.down_size, self.n_embd, bias=False)
        self.lora_B = nn.Linear(self.n_embd, self.down_size, bias=False)

        # ---- EWC (shared-adapter) Fisher estimation hooks ----
        # When ewc_fisher_mode is True, forward builds the full update
        # delta_w = A_s @ B_s, calls retain_grad on it, and routes the forward
        # through it so that delta_w.grad gives d(CE)/d(Delta_W_s). These are
        # only used during SharedAdapterEWC.estimate_fisher and never register
        # delta_w as a trainable parameter.
        self.ewc_fisher_mode = False
        self.ewc_delta_w = None

        if self.random_orth:
            random_matrix = torch.rand(self.n_embd, self.down_size)
            q, r = torch.linalg.qr(random_matrix)
            with torch.no_grad():
                self.lora_B.weight.copy_(q.T)
            scaling_factor = 1.  # You can adjust this value if needed
            self.lora_B.weight.data *= scaling_factor
        else:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))

        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.zeros_(self.lora_A.weight)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.ewc_fisher_mode:
            # Build the full low-rank update delta_w = A_s @ B_s in the
            # [n_embd, n_embd] space and route the forward through it so the
            # diagonal Fisher can be read from delta_w.grad. lora_A is trainable
            # and lora_B is frozen, so gradients still flow back to A_s.
            delta_w = self.lora_A.weight @ self.lora_B.weight  # [n_embd, n_embd]
            delta_w.retain_grad()
            self.ewc_delta_w = delta_w
            out = F.linear(x, delta_w)  # x @ delta_w^T == lora_A(lora_B(x))
            return out
        inter_x = self.lora_B(x)
        out = self.lora_A(inter_x)
        return out


class SDLoRAAdapter(nn.Module):
    is_sd_lora_adapter = True

    def __init__(self,
                 config=None,
                 rank=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.config = config
        self.rank = int(rank if rank is not None else config.ffn_num)
        self.direction_norm_eps = float(getattr(config, "direction_norm_eps", 1e-6))
        self.normalize_direction = bool(getattr(config, "normalize_direction", True))
        self.specific_lora_init_scale = float(getattr(config, "specific_lora_init_scale", 1e-3))
        self.cache_old_directions = bool(getattr(config, "cache_old_directions", True))
        self.cache_device = getattr(config, "cache_device", "cuda")
        if self.cache_device not in ("cuda", "cpu"):
            raise ValueError("sd_lora.cache_device must be 'cuda' or 'cpu', got {}".format(self.cache_device))
        self.combine_directions_before_linear = bool(
            getattr(config, "combine_directions_before_linear", False)
        )
        self.directions = nn.ModuleList()
        self.register_buffer(
            "cached_old_direction_weights",
            torch.empty(0),
            persistent=False,
        )
        self._cached_old_direction_count = 0
        self._cache_warning_emitted = set()
        self._adapter_kwargs = dict(
            config=config,
            dropout=dropout,
            init_option=init_option,
            adapter_scalar=adapter_scalar,
            adapter_layernorm_option=adapter_layernorm_option,
        )
        self.add_direction(self.rank)

    def add_direction(self, rank):
        direction = Adapter_lora(
            bottleneck=int(rank),
            **self._adapter_kwargs,
        )
        with torch.no_grad():
            nn.init.normal_(direction.lora_A.weight, mean=0.0, std=self.specific_lora_init_scale)
        direction.requires_grad_(True)
        self.directions.append(direction)
        return direction

    def freeze_all_directions(self):
        for direction in self.directions:
            direction.requires_grad_(False)

    def set_trainable_current_direction(self):
        self.freeze_all_directions()
        if len(self.directions) > 0:
            self.directions[-1].requires_grad_(True)

    def direction_ranks(self):
        return [direction.lora_B.weight.shape[0] for direction in self.directions]

    def num_cached_directions(self):
        return int(self._cached_old_direction_count)

    def current_direction_cached(self):
        return len(self.directions) > 0 and self.num_cached_directions() >= len(self.directions)

    def clear_cached_old_directions(self):
        device = self.cached_old_direction_weights.device
        dtype = self.cached_old_direction_weights.dtype
        self.cached_old_direction_weights = torch.empty(0, device=device, dtype=dtype)
        self._cached_old_direction_count = 0
        self._cache_warning_emitted.clear()

    def drop_direction_cache(self):
        """Free the normalized-direction cache on a frozen snapshot adapter.

        Snapshots in adapter_list / old_adapter_list are deep copies and would
        otherwise each carry a full [num_old, d, d] cache copy -> O(T^2) GPU
        memory across tasks. After dropping, forward falls back to recomputing
        normalize(A@B) on the fly, which is numerically identical (the cache
        stored exactly that) and cheap. The fallback warning is pre-silenced
        because the fallback is intentional here."""
        self.clear_cached_old_directions()
        self._cache_warning_emitted.update(range(len(self.directions)))

    def _cache_target_device(self):
        parameter_device = next(self.parameters()).device
        if self.cache_device == "cpu":
            return torch.device("cpu")
        if parameter_device.type == "cuda":
            return parameter_device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return parameter_device

    def _direction_weight(self, direction):
        update_weight = direction.lora_A.weight @ direction.lora_B.weight
        return update_weight

    def _normalize_direction_weight(self, update_weight):
        if not self.normalize_direction:
            # normalize_direction=False -> raw direction D_k = A_k B_k (e.g. baseline).
            return update_weight
        denom = torch.norm(update_weight, p="fro") + self.direction_norm_eps
        return update_weight / denom

    def _normalized_direction_weight(self, direction):
        return self._normalize_direction_weight(self._direction_weight(direction))

    def cache_frozen_old_directions(self, include_current=False):
        if not self.cache_old_directions:
            self.clear_cached_old_directions()
            return None

        limit = len(self.directions) if include_current else max(len(self.directions) - 1, 0)
        cached_weights = []
        target_device = self._cache_target_device()
        with torch.no_grad():
            for direction_index in range(limit):
                direction = self.directions[direction_index]
                if any(parameter.requires_grad for parameter in direction.parameters()):
                    break
                update_weight = self._normalized_direction_weight(direction).detach()
                cached_weights.append(update_weight.to(device=target_device))

        if len(cached_weights) == 0:
            self.clear_cached_old_directions()
            return None

        self.cached_old_direction_weights = torch.stack(cached_weights, dim=0).detach()
        self._cached_old_direction_count = int(self.cached_old_direction_weights.shape[0])
        self._cache_warning_emitted.clear()
        return {
            "count": self._cached_old_direction_count,
            "shape": tuple(self.cached_old_direction_weights.shape),
            "device": str(self.cached_old_direction_weights.device),
            "dtype": str(self.cached_old_direction_weights.dtype),
            "requires_grad": bool(self.cached_old_direction_weights.requires_grad),
        }

    def _cached_direction_weight_for_forward(self, direction_index, x, active_directions):
        if not self.cache_old_directions:
            return None

        is_current_direction = direction_index == active_directions - 1
        if is_current_direction:
            return None

        if (
            self.cached_old_direction_weights.numel() > 0
            and direction_index < self.num_cached_directions()
        ):
            return self.cached_old_direction_weights[direction_index].to(
                device=x.device,
                dtype=x.dtype,
                non_blocking=True,
            )

        if direction_index not in self._cache_warning_emitted:
            logging.warning(
                "SD-LoRA old direction cache missing; falling back to A@B. "
                "direction_index=%s active_directions=%s cached_directions=%s",
                direction_index,
                active_directions,
                self.num_cached_directions(),
            )
            self._cache_warning_emitted.add(direction_index)
        return None

    def _direction_weight_for_forward(self, direction_index, x, active_directions):
        cached_weight = self._cached_direction_weight_for_forward(
            direction_index,
            x,
            active_directions,
        )
        if cached_weight is not None:
            return cached_weight
        return self._normalized_direction_weight(self.directions[direction_index]).to(
            device=x.device,
            dtype=x.dtype,
        )

    def orthogonality_loss(self):
        """|<D_t_hat, D_k_hat>_F| between the current (trainable) direction and
        every frozen old direction, averaged. Directions are Frobenius-normalized,
        so each term lies in [0, 1]. Returns None when there is no old direction.

        Old directions come from cached_old_direction_weights when available
        (already normalized + detached); otherwise they are recomputed without
        grad. Only the current direction receives gradient."""
        if len(self.directions) < 2:
            return None
        current_weight = self._normalized_direction_weight(self.directions[-1])
        total = None
        num_old = len(self.directions) - 1
        for k in range(num_old):
            if self.cached_old_direction_weights.numel() > 0 and k < self.num_cached_directions():
                old_weight = self.cached_old_direction_weights[k].to(
                    device=current_weight.device, dtype=current_weight.dtype
                )
            else:
                with torch.no_grad():
                    old_weight = self._normalized_direction_weight(self.directions[k]).detach()
            dot = torch.abs((current_weight * old_weight).sum())
            total = dot if total is None else total + dot
        return total / num_old

    def forward(self, x, direction_scale):
        if direction_scale is None:
            raise ValueError("SDLoRAAdapter requires a direction_scale tensor.")
        active_directions = min(len(self.directions), direction_scale.shape[0])
        if self.combine_directions_before_linear:
            combined_weight = None
            for direction_index in range(active_directions):
                update_weight = self._direction_weight_for_forward(direction_index, x, active_directions)
                scale = direction_scale[direction_index].to(device=x.device, dtype=x.dtype)
                scaled_weight = scale * update_weight
                combined_weight = scaled_weight if combined_weight is None else combined_weight + scaled_weight
            if combined_weight is None:
                return torch.zeros_like(x)
            return F.linear(x, combined_weight)

        out = torch.zeros_like(x)
        for direction_index in range(active_directions):
            update_weight = self._direction_weight_for_forward(direction_index, x, active_directions)
            direction_out = F.linear(x, update_weight)
            scale = direction_scale[direction_index].to(device=direction_out.device, dtype=direction_out.dtype)
            out = out + scale * direction_out
        return out


class Attention_lora(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., msa = [0,0,0]):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.ffn_option = 'parallel'
        self.msa = msa


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def _adapter_forward(self, adapter, x, direction_scale=None):
        if getattr(adapter, "is_sd_lora_adapter", False):
            return adapter(x, direction_scale)
        return adapter(x)


    def forward(self, x, adapt=None, prompt = None, rank_prompt = None, block_weight = None, direction_scale=None):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if adapt is not None:
            if block_weight is not None:
                block_weight = block_weight
            else:
                block_weight = torch.ones(3, device=x.device, dtype=x.dtype)
            if self.msa[0] == 1:
                adapt_x = self._adapter_forward(adapt[0], x, direction_scale)
                q += block_weight[0] * adapt_x
            if self.msa[1] == 1:
                adapt_x = self._adapter_forward(adapt[1], x, direction_scale)
                k += block_weight[1] * adapt_x
            if self.msa[2] == 1:
                adapt_x = self._adapter_forward(adapt[2], x, direction_scale)
                v += block_weight[2] * adapt_x


        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)


        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.msa_adapt = True
        self.norm1 = norm_layer(dim)
        self.attn = Attention_lora(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, msa = config.msa)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)



    # prompt and rank_prmopt can be considerred as potential future improvements by levergaing additional prompt information, but is not implemented in this work
    def forward(self, x, adapt=None, prompt=None, rank_prompt=None, block_weight=None, direction_scale=None):
        if self.msa_adapt:
            x = x + self.drop_path(
                self.attn(self.norm1(x), adapt, prompt, rank_prompt, block_weight, direction_scale))
            residual = x
            x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
            x = self.drop_path(self.mlp_drop(self.fc2(x)))
            x = residual + x
        return x



class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()

        self.tuning_config = tuning_config
        if self.tuning_config.ffn_adapt:
            print("I'm using ViT with adapters.")
        else:
            print("I'm using ViT without adapters.")
            self.maskout_block = []
        self.adapt_msa = True
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.msa_adapt = self.tuning_config.msa_adapt
        self.use_distillation = self.tuning_config.use_distillation
        self.use_block_weight = self.tuning_config.use_block_weight
        self.direction_scale_init = float(getattr(self.tuning_config, "direction_scale_init", 0.05))
        self.block_weight_norm_eps = float(getattr(self.tuning_config, "block_weight_norm_eps", 1e-6))
        self.block_weight_normalization = getattr(self.tuning_config, "block_weight_normalization", "mean_l1")
        # SD-LoRA-RR task-specific adapter settings
        self.sd_lora_enable = bool(getattr(self.tuning_config, "sd_lora_enable", True))
        self.sd_lora_variant = getattr(self.tuning_config, "sd_lora_variant", "rr")
        self.sd_lora_r0 = int(getattr(self.tuning_config, "sd_lora_r0", self.tuning_config.ffn_num))
        self.sd_lora_r_min = int(getattr(self.tuning_config, "sd_lora_r_min", 1))
        self.sd_lora_rank_decay = float(getattr(self.tuning_config, "sd_lora_rank_decay", 1.0))
        self.normalize_direction = bool(getattr(self.tuning_config, "normalize_direction", True))
        self.alpha_mode = getattr(self.tuning_config, "alpha_mode", "task_conditioned")
        self.train_old_alpha = bool(getattr(self.tuning_config, "train_old_alpha", False))
        self.cache_old_directions = bool(getattr(self.tuning_config, "cache_old_directions", True))
        self.cache_device = getattr(self.tuning_config, "cache_device", "cuda")
        if self.cache_device not in ("cuda", "cpu"):
            raise ValueError("sd_lora.cache_device must be 'cuda' or 'cpu', got {}".format(self.cache_device))
        self.combine_directions_before_linear = bool(
            getattr(self.tuning_config, "combine_directions_before_linear", False)
        )
        self.current_task_index = 0

        if self.msa_adapt:
            self.msa = self.tuning_config.msa
        self.general_pos = self.tuning_config.general_pos
        self.specific_pos = self.tuning_config.specific_pos
        # B7(a): original CL-LoRA inference semantics — every task branch uses
        # the CURRENT shared adapters instead of its per-task snapshot, which
        # lets the shared prefix be computed once per batch (O(l+(N-l)T)).
        self.eval_shared_current = bool(
            getattr(self.tuning_config, "eval_shared_current", False))

        self.adapt_pos = self.general_pos+ self.specific_pos
        self.adapt_pos = sorted(self.adapt_pos)


        if self.use_distillation:
            self.old_adapter_list = nn.ModuleList()

        if self.use_block_weight:
            self.block_weight_list = nn.ParameterList()
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specific_pos)))
            nn.init.uniform_(self.block_weight, .5, 1.5)

        self.direction_scale_list = nn.ParameterList()
        self.direction_scale = nn.Parameter(
            torch.full((len(self.specific_pos), 1), self.direction_scale_init),
            requires_grad=self.sd_lora_enable,
        )
        self.current_specific_rank = self.get_specific_lora_rank(0)
        self.register_buffer(
            "specific_lora_rank_history_buffer",
            torch.tensor([self.current_specific_rank], dtype=torch.long),
        )

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        ######## Adapter begins #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.config = tuning_config
        self._device = tuning_config._device
        self.adapter_list = nn.ModuleList()
        self.adapter_pos_list = []
        self.cur_adapter = nn.ModuleList()
        if self.msa_adapt:
            self.get_new_adapter_initial_msa()

    def get_specific_lora_rank(self, task_id):
        # SD-LoRA-RR rank schedule for the task-specific adapter.
        # task_id is zero-based: task_id=0 -> first task (t=1).
        if not self.sd_lora_enable:
            # Experiment A baseline: original CL-LoRA, one fixed-rank direction per task.
            return self.sd_lora_r0
        if self.sd_lora_variant == "fixed":
            # Experiment B: fixed-rank SD-LoRA, every new direction keeps r0.
            return self.sd_lora_r0
        if self.sd_lora_variant == "rr":
            # Experiment C/D: r_t = max(r_min, floor(r0 * decay^(t-1)))
            decayed = int(math.floor(self.sd_lora_r0 * (self.sd_lora_rank_decay ** task_id)))
            return max(self.sd_lora_r_min, decayed)

        # Legacy milestone-based schedule fallback (kept for backward compatibility).
        schedule = getattr(self.tuning_config, "specific_lora_rank_schedule", None)
        default_rank = int(self.tuning_config.ffn_num)
        if not schedule:
            return default_rank

        milestones = list(schedule.get("milestones", []))
        ranks = list(schedule.get("ranks", [default_rank]))
        if len(ranks) != len(milestones) + 1:
            raise ValueError("specific_lora_rank_schedule.ranks must have len(milestones) + 1 entries.")

        rank = int(ranks[0])
        if len(milestones) > 0 and all(isinstance(m, float) and m <= 1.0 for m in milestones):
            nb_tasks = max(int(getattr(self.tuning_config, "nb_tasks", 1)), 1)
            denominator = max(nb_tasks - 1, 1)
            progress = float(task_id) / denominator
            for milestone, next_rank in zip(milestones, ranks[1:]):
                if progress >= float(milestone):
                    rank = int(next_rank)
        else:
            for milestone, next_rank in zip(milestones, ranks[1:]):
                if task_id >= int(milestone):
                    rank = int(next_rank)
        return rank

    def _new_direction_scale(self, previous_scale=None):
        if previous_scale is None:
            values = torch.full((len(self.specific_pos), 1), self.direction_scale_init)
        else:
            inherited = previous_scale.detach().clone()
            if inherited.dim() == 1:
                inherited = inherited.unsqueeze(0).repeat(len(self.specific_pos), 1)
            new_value = inherited.new_full((inherited.shape[0], 1), self.direction_scale_init)
            values = torch.cat([inherited, new_value], dim=1)
        return nn.Parameter(values, requires_grad=self.sd_lora_enable)

    def _new_block_weight(self):
        block_weight = nn.Parameter(torch.randn(3, len(self.specific_pos)))
        nn.init.uniform_(block_weight, .5, 1.5)
        return block_weight

    def _set_specific_lora_rank_history(self, ranks):
        device = self.specific_lora_rank_history_buffer.device
        self.specific_lora_rank_history_buffer = torch.tensor(
            [int(rank) for rank in ranks],
            dtype=torch.long,
            device=device,
        )

    def _specific_lora_rank_history(self):
        return [int(rank) for rank in self.specific_lora_rank_history_buffer.detach().cpu().tolist()]

    def normalize_block_weight(self, block_weight):
        if self.block_weight_normalization in [None, "none", "None"]:
            return block_weight
        if self.block_weight_normalization == "mean_l1":
            denom = block_weight.abs().mean() + self.block_weight_norm_eps
        elif self.block_weight_normalization == "l2":
            denom = torch.norm(block_weight.flatten(), p=2) + self.block_weight_norm_eps
        elif self.block_weight_normalization == "l1":
            denom = torch.norm(block_weight.flatten(), p=1) + self.block_weight_norm_eps
        else:
            raise ValueError("Unsupported block_weight_normalization: {}".format(self.block_weight_normalization))
        return block_weight / denom

    def _get_block_weight_column(self, block_weight, pos_spec):
        return self.normalize_block_weight(block_weight)[:, pos_spec]

    def _snapshot_parameter(self, parameter):
        return nn.Parameter(parameter.detach().clone(), requires_grad=False)

    def _set_specific_trainable_state(self):
        for block_idx in self.specific_pos:
            pos = self.adapt_pos.index(block_idx)
            for adapter in self.cur_adapter[pos]:
                if getattr(adapter, "is_sd_lora_adapter", False):
                    adapter.set_trainable_current_direction()

    def _apply_alpha_mode_requires_grad(self):
        if not self.sd_lora_enable:
            self.direction_scale.requires_grad = False
            for parameter in self.direction_scale_list:
                parameter.requires_grad = False
            return
        # The current-task routing alpha (self.direction_scale) is always trainable:
        #   - global mode: it holds the single shared alpha_1..alpha_t (all trainable);
        #   - task_conditioned mode: it is the current branch routing alpha_{t,1:t}.
        self.direction_scale.requires_grad = True
        # Old-task routing snapshots stay frozen unless train_old_alpha is set, so the
        # old task's prototype space is not perturbed by later tasks.
        for parameter in self.direction_scale_list:
            parameter.requires_grad = bool(self.train_old_alpha)

    def _select_direction_scale_for_block(self, direction_scale, pos_spec):
        if direction_scale is None:
            return None
        if direction_scale.dim() == 1:
            return direction_scale
        return direction_scale[pos_spec]

    def _inference_direction_scale(self, task_index, pos_spec):
        if self.alpha_mode == "global":
            # Global alpha: all task branches share the latest routing vector prefix.
            return self._select_direction_scale_for_block(
                self.direction_scale,
                pos_spec,
            )[: task_index + 1]
        # task_conditioned: each branch uses its own saved (frozen) routing snapshot.
        return self._select_direction_scale_for_block(
            self.direction_scale_list[task_index],
            pos_spec,
        )

    def _get_general_adapter_for_snapshot(self, task_index, block_index):
        pos = self.adapt_pos.index(block_index)
        if self.eval_shared_current:
            return self.cur_adapter[pos]
        if self.use_distillation and task_index < len(self.old_adapter_list):
            return self.old_adapter_list[task_index][pos]
        return self.cur_adapter[pos]

    def _current_sd_lora_adapters(self):
        adapters = []
        if not self.msa_adapt:
            return adapters
        for block_idx in self.specific_pos:
            pos = self.adapt_pos.index(block_idx)
            for adapter in self.cur_adapter[pos]:
                if getattr(adapter, "is_sd_lora_adapter", False):
                    adapters.append(adapter)
        return adapters

    def sd_lora_direction_orth_loss(self):
        """Mean orthogonality loss of the current SD-LoRA direction against all
        frozen old directions, over every task-specific adapter. None if no old
        directions exist yet (e.g. task 0)."""
        losses = []
        for adapter in self._current_sd_lora_adapters():
            adapter_loss = adapter.orthogonality_loss()
            if adapter_loss is not None:
                losses.append(adapter_loss)
        if len(losses) == 0:
            return None
        return torch.stack(losses).mean()

    def cache_current_old_directions(self):
        if not self.sd_lora_enable or not self.cache_old_directions:
            return []
        summaries = []
        for adapter in self._current_sd_lora_adapters():
            summary = adapter.cache_frozen_old_directions(include_current=False)
            if summary is not None:
                summaries.append(summary)
        return summaries

    def log_sd_lora_cache_state(self):
        adapters = self._current_sd_lora_adapters()
        num_cached_directions = sum(adapter.num_cached_directions() for adapter in adapters)
        current_direction_cached = any(adapter.current_direction_cached() for adapter in adapters)
        logging.info(
            "[SD-LoRA Cache]\n"
            "cache_old_directions: %s\n"
            "cache_device: %s\n"
            "num_cached_directions: %s\n"
            "current_direction_cached: %s\n"
            "combine_directions_before_linear: %s",
            self.cache_old_directions,
            self.cache_device,
            num_cached_directions,
            current_direction_cached,
            self.combine_directions_before_linear,
        )

    def log_sd_lora_cache_saved(self, task_id, summaries):
        if len(summaries) == 0:
            return
        first = summaries[0]
        logging.info(
            "[SD-LoRA Cache Saved]\n"
            "task_id: %s\n"
            "num_layers: %s\n"
            "num_positions: %s\n"
            "cached_weight_shape: %s\n"
            "device: %s\n"
            "dtype: %s\n"
            "requires_grad: %s",
            task_id,
            len(self.specific_pos),
            len(summaries),
            first["shape"],
            first["device"],
            first["dtype"],
            first["requires_grad"],
        )

    def log_sd_lora_rr_state(self):
        direction_ranks = []
        current_direction_params = 0
        old_directions_frozen = True
        current_adapter = None
        if self.msa_adapt and len(self.specific_pos) > 0:
            current_adapter = self.cur_adapter[self.adapt_pos.index(self.specific_pos[0])][0]
            if getattr(current_adapter, "is_sd_lora_adapter", False):
                direction_ranks = current_adapter.direction_ranks()
                # parameter count of the newly added (current) direction
                current_direction_params = sum(
                    p.numel() for p in current_adapter.directions[-1].parameters()
                )
                # whether every direction except the last is frozen
                for direction in current_adapter.directions[:-1]:
                    if any(p.requires_grad for p in direction.parameters()):
                        old_directions_frozen = False
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        use_orth = bool(getattr(self.tuning_config, "use_orthogonal_constraint", False))
        old_alpha_grad = [bool(p.requires_grad) for p in self.direction_scale_list]
        logging.info(
            "SD-LoRA-RR | task_id=%s enable=%s variant=%s alpha_mode=%s train_old_alpha=%s "
            "new_rank=%s direction_ranks=%s new_direction_params=%s old_directions_frozen=%s "
            "alpha_shape=%s alpha_requires_grad=%s old_alpha_requires_grad=%s "
            "normalize_direction=%s use_block_weight=%s use_orth_loss=%s "
            "total_params=%s trainable_params=%s",
            self.current_task_index,
            self.sd_lora_enable,
            self.sd_lora_variant,
            self.alpha_mode,
            self.train_old_alpha,
            self.current_specific_rank,
            direction_ranks,
            current_direction_params,
            old_directions_frozen,
            tuple(self.direction_scale.shape),
            bool(self.direction_scale.requires_grad),
            old_alpha_grad,
            self.normalize_direction,
            self.use_block_weight,
            use_orth,
            total_params,
            trainable_params,
        )
        self.log_sd_lora_cache_state()

    def _snapshot_indices_in_state_dict(self, state_dict, prefix):
        indices = set()
        pattern = re.compile(r"^{}(\d+)".format(re.escape(prefix)))
        for key in state_dict.keys():
            match = pattern.match(key)
            if match:
                indices.add(int(match.group(1)))
        return sorted(indices)

    def _direction_ranks_from_state_dict(self, state_dict, prefix):
        ranks = {}
        pattern = re.compile(r"^{}directions\.(\d+)\.lora_B\.weight$".format(re.escape(prefix)))
        for key, value in state_dict.items():
            match = pattern.match(key)
            if match:
                ranks[int(match.group(1))] = int(value.shape[0])
        return [rank for _, rank in sorted(ranks.items())]

    def _make_sd_adapter_from_ranks(self, ranks, trainable_current):
        if len(ranks) == 0:
            ranks = [self.current_specific_rank]
        adapter = SDLoRAAdapter(
            self.config,
            rank=ranks[0],
            dropout=0.0,
            init_option=self.config.ffn_adapter_init_option,
            adapter_scalar=self.config.ffn_adapter_scalar,
            adapter_layernorm_option=self.config.ffn_adapter_layernorm_option,
        ).to(self._device)
        while len(adapter.directions) < len(ranks):
            adapter.add_direction(ranks[len(adapter.directions)])
        if trainable_current:
            adapter.set_trainable_current_direction()
        else:
            adapter.freeze_all_directions()
        return adapter

    def _make_standard_adapter_from_state_dict(self, state_dict, prefix, trainable):
        rank = int(state_dict[prefix + "lora_B.weight"].shape[0])
        adapter = Adapter_lora(
            self.config,
            dropout=0.0,
            bottleneck=rank,
            init_option=self.config.ffn_adapter_init_option,
            adapter_scalar=self.config.ffn_adapter_scalar,
            adapter_layernorm_option=self.config.ffn_adapter_layernorm_option,
        ).to(self._device)
        adapter.requires_grad_(trainable)
        return adapter

    def _rebuild_current_adapters_from_state_dict(self, state_dict):
        if "direction_scale" in state_dict:
            self.direction_scale = nn.Parameter(torch.zeros_like(state_dict["direction_scale"]))
            self.current_task_index = max(int(state_dict["direction_scale"].shape[-1]) - 1, 0)

        if "block_weight" in state_dict:
            self.block_weight = nn.Parameter(torch.zeros_like(state_dict["block_weight"]))

        rank_history = None
        for block_idx in self.specific_pos:
            pos = self.adapt_pos.index(block_idx)
            temp_adapter = nn.ModuleList()
            for msa_idx, msa_enabled in enumerate(self.msa):
                if msa_enabled == 1:
                    ranks = self._direction_ranks_from_state_dict(
                        state_dict,
                        "cur_adapter.{}.{}.".format(pos, msa_idx),
                    )
                    adapter = self._make_sd_adapter_from_ranks(ranks, trainable_current=True)
                    if rank_history is None:
                        rank_history = ranks
                else:
                    adapter = nn.Identity()
                temp_adapter.append(adapter)
            self.cur_adapter[pos] = temp_adapter

        if rank_history:
            self._set_specific_lora_rank_history(rank_history)
            self.current_specific_rank = rank_history[-1]

        if "specific_lora_rank_history_buffer" in state_dict:
            self.specific_lora_rank_history_buffer = torch.zeros_like(
                state_dict["specific_lora_rank_history_buffer"]
            )

    def _rebuild_snapshot_adapters_from_state_dict(self, state_dict):
        adapter_indices = self._snapshot_indices_in_state_dict(state_dict, "adapter_list.")
        self.adapter_list = nn.ModuleList()
        for snapshot_idx in adapter_indices:
            snapshot_adapters = []
            for spec_idx in range(len(self.specific_pos)):
                temp_adapter = nn.ModuleList()
                for msa_idx, msa_enabled in enumerate(self.msa):
                    if msa_enabled == 1:
                        ranks = self._direction_ranks_from_state_dict(
                            state_dict,
                            "adapter_list.{}.{}.{}.".format(snapshot_idx, spec_idx, msa_idx),
                        )
                        adapter = self._make_sd_adapter_from_ranks(ranks, trainable_current=False)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)
                snapshot_adapters.append(temp_adapter)
            self.adapter_list.append(nn.ModuleList(snapshot_adapters).requires_grad_(False))

    def _rebuild_old_adapters_from_state_dict(self, state_dict):
        if not self.use_distillation:
            return
        old_indices = self._snapshot_indices_in_state_dict(state_dict, "old_adapter_list.")
        self.old_adapter_list = nn.ModuleList()
        for snapshot_idx in old_indices:
            snapshot = nn.ModuleList()
            for pos, block_idx in enumerate(self.adapt_pos):
                temp_adapter = nn.ModuleList()
                for msa_idx, msa_enabled in enumerate(self.msa):
                    if msa_enabled == 1:
                        prefix = "old_adapter_list.{}.{}.{}.".format(snapshot_idx, pos, msa_idx)
                        if block_idx in self.specific_pos:
                            ranks = self._direction_ranks_from_state_dict(state_dict, prefix)
                            adapter = self._make_sd_adapter_from_ranks(ranks, trainable_current=False)
                        else:
                            adapter = self._make_standard_adapter_from_state_dict(state_dict, prefix, trainable=False)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)
                snapshot.append(temp_adapter)
            self.old_adapter_list.append(snapshot.requires_grad_(False))

    def _rebuild_parameter_snapshots_from_state_dict(self, state_dict):
        self.block_weight_list = nn.ParameterList()
        for snapshot_idx in self._snapshot_indices_in_state_dict(state_dict, "block_weight_list."):
            key = "block_weight_list.{}".format(snapshot_idx)
            self.block_weight_list.append(nn.Parameter(torch.zeros_like(state_dict[key]), requires_grad=False))

        self.direction_scale_list = nn.ParameterList()
        for snapshot_idx in self._snapshot_indices_in_state_dict(state_dict, "direction_scale_list."):
            key = "direction_scale_list.{}".format(snapshot_idx)
            self.direction_scale_list.append(nn.Parameter(torch.zeros_like(state_dict[key]), requires_grad=False))

    def _rebuild_sd_lora_rr_from_state_dict(self, state_dict):
        if "direction_scale" not in state_dict:
            return
        self._rebuild_current_adapters_from_state_dict(state_dict)
        self._rebuild_snapshot_adapters_from_state_dict(state_dict)
        self._rebuild_old_adapters_from_state_dict(state_dict)
        self._rebuild_parameter_snapshots_from_state_dict(state_dict)

    def _restore_requires_grad_after_load(self):
        for parameter in self.adapter_list.parameters():
            parameter.requires_grad = False
        if hasattr(self, "old_adapter_list"):
            for parameter in self.old_adapter_list.parameters():
                parameter.requires_grad = False
        if hasattr(self, "block_weight_list"):
            for parameter in self.block_weight_list:
                parameter.requires_grad = False
        if hasattr(self, "block_weight"):
            self.block_weight.requires_grad = True
        self._apply_alpha_mode_requires_grad()
        self._set_specific_trainable_state()

    def load_state_dict(self, state_dict, strict=True):
        if strict and "direction_scale" not in state_dict:
            raise RuntimeError(
                "This checkpoint does not contain SD-LoRA-RR state. "
                "It appears to be a pre-SD-LoRA-RR CL-LoRA checkpoint and cannot restore "
                "task-specific direction scales or adapter snapshots."
            )
        self._rebuild_sd_lora_rr_from_state_dict(state_dict)
        incompatible_keys = super().load_state_dict(state_dict, strict=strict)
        if "direction_scale" in state_dict:
            self._restore_requires_grad_after_load()
        return incompatible_keys

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist           

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True


    def get_new_adapter_initial_msa(self):
        config = self.config
        if config.ffn_adapt:
            for i in range(len(self.adapt_pos)):
                temp_adapter = nn.ModuleList()
                for j in self.msa:
                    if j == 1:
                        if self.adapt_pos[i] in self.specific_pos and self.sd_lora_enable:
                            adapter = SDLoRAAdapter(self.config,
                                                   rank=self.current_specific_rank,
                                                   dropout=0.0,
                                                   init_option=config.ffn_adapter_init_option,
                                                   adapter_scalar=config.ffn_adapter_scalar,
                                                   adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                   ).to(self._device)
                        else:
                            adapter_rank = self.current_specific_rank if self.adapt_pos[i] in self.specific_pos else config.ffn_num
                            adapter = Adapter_lora(self.config, dropout=0.0, bottleneck=adapter_rank,
                                                    init_option=config.ffn_adapter_init_option,
                                                    adapter_scalar=config.ffn_adapter_scalar,
                                                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                    ).to(self._device)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)

                self.cur_adapter.append(temp_adapter)
            self.cur_adapter.requires_grad_(True)
            self._set_specific_trainable_state()

    def get_new_adapter_msa(self):
        config = self.config

        if config.ffn_adapt:
            for i in range(len(self.specific_pos)):
                pos = self.adapt_pos.index(self.specific_pos[i])
                temp_adapter = nn.ModuleList()
                for j in self.msa:
                    if j == 1:
                        previous_adapter = self.cur_adapter[pos][len(temp_adapter)]
                        if self.sd_lora_enable and getattr(previous_adapter, "is_sd_lora_adapter", False):
                            # SD-LoRA: accumulate directions, freeze old ones, train new one.
                            adapter = copy.deepcopy(previous_adapter).to(self._device)
                            adapter.freeze_all_directions()
                            adapter.add_direction(self.current_specific_rank)
                            adapter.set_trainable_current_direction()
                        elif self.sd_lora_enable:
                            adapter = SDLoRAAdapter(self.config,
                                                   rank=self.current_specific_rank,
                                                   dropout=0.0,
                                                   init_option=config.ffn_adapter_init_option,
                                                   adapter_scalar=config.ffn_adapter_scalar,
                                                   adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                   ).to(self._device)
                        else:
                            adapter = Adapter_lora(self.config,
                                                   dropout=0.0,
                                                   bottleneck=self.current_specific_rank,
                                                   init_option=config.ffn_adapter_init_option,
                                                   adapter_scalar=config.ffn_adapter_scalar,
                                                   adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                   ).to(self._device)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)
                self.cur_adapter[pos] = temp_adapter

            if len(self.specific_pos) < 12:
                self.cur_adapter.requires_grad_(True)

                for i in self.adapt_pos:
                    if i in self.general_pos:
                        pos = self.adapt_pos.index(i)
                        for j in range(len(self.msa)):
                            if self.msa[j] == 1:
                                self.cur_adapter[pos][j].lora_B.requires_grad_(False)
                self._set_specific_trainable_state()

    def _drop_snapshot_direction_caches(self, module):
        for submodule in module.modules():
            if getattr(submodule, "is_sd_lora_adapter", False):
                submodule.drop_direction_cache()

    def add_adapter_to_list(self):
        completed_task_id = self.current_task_index
        temp_adapter = []
        for i in range(len(self.specific_pos)):
            temp_pos = self.adapt_pos.index(self.specific_pos[i])
            temp_adapter.append(copy.deepcopy(self.cur_adapter[temp_pos].requires_grad_(False)))
        self.adapter_list.append(nn.ModuleList(temp_adapter))
        # deepcopy duplicated the [num_old, d, d] direction cache into the
        # snapshot -> O(T^2) GPU memory across tasks; drop it (fallback is
        # numerically identical).
        self._drop_snapshot_direction_caches(self.adapter_list[-1])

        if self.use_block_weight:
            self.block_weight_list.append(self._snapshot_parameter(self.block_weight))
            self.block_weight = self._new_block_weight()

        self.direction_scale_list.append(self._snapshot_parameter(self.direction_scale))
        previous_direction_scale = self.direction_scale
        self.current_task_index += 1
        self.current_specific_rank = self.get_specific_lora_rank(self.current_task_index)
        rank_history = self._specific_lora_rank_history()
        rank_history.append(self.current_specific_rank)
        self._set_specific_lora_rank_history(rank_history)
        self.direction_scale = self._new_direction_scale(previous_direction_scale)


        self.adapter_pos_list.append(self.adapt_pos)

        if self.use_distillation:
            snapshot = copy.deepcopy(self.cur_adapter).requires_grad_(False)
            # KD/inference only ever read the general_pos (shared) entries of
            # old_adapter_list (see _get_general_adapter_for_snapshot /
            # forward_general_cls); the task-specific entries — whose
            # SD-LoRA directions grow with the task count — are dead weight,
            # so keep only the shared entries indexable.
            for block_index in self.specific_pos:
                snapshot[self.adapt_pos.index(block_index)] = nn.Identity()
            self._drop_snapshot_direction_caches(snapshot)
            self.old_adapter_list.append(snapshot)
        if self.msa_adapt:
            self.get_new_adapter_msa()
        self._apply_alpha_mode_requires_grad()
        cache_summaries = self.cache_current_old_directions()
        self.log_sd_lora_cache_saved(completed_task_id, cache_summaries)
        self.log_sd_lora_rr_state()

    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            rank_prompt = None
            prompt = None

            if self.config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)

            if self.config.ffn_adapt:
                if idx in self.adapt_pos:
                    pos = self.adapt_pos.index(idx)
                    block_weight = None
                    if idx in self.specific_pos:
                        pos_spec = self.specific_pos.index(idx)
                        if self.use_block_weight:
                            block_weight = self._get_block_weight_column(self.block_weight, pos_spec)
                        x = blk(x, self.cur_adapter[pos], prompt, rank_prompt,
                                block_weight=block_weight,
                                direction_scale=self._select_direction_scale_for_block(self.direction_scale, pos_spec))
                    else:
                        x = blk(x, self.cur_adapter[pos], prompt, rank_prompt, block_weight=None)
                else:
                    x = blk(x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            else:
                x = blk(x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            if self.config.vpt_on:
                x = x[:, self.config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_test(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)

        features = []

        # B7(a) fast path: with eval_shared_current every branch runs the same
        # (current) shared adapters in the prefix blocks, so compute that
        # prefix once per batch instead of once per branch.
        shared_prefix_x = None
        prefix_boundary = 0
        if (
            self.eval_shared_current
            and self.config.ffn_adapt
            and len(self.specific_pos) > 0
            and (len(self.general_pos) == 0 or max(self.general_pos) < min(self.specific_pos))
        ):
            prefix_boundary = min(self.specific_pos)
            xp = x_init
            for j in range(prefix_boundary):
                if j in self.adapt_pos:
                    xp = self.blocks[j](xp, self.cur_adapter[self.adapt_pos.index(j)])
                else:
                    xp = self.blocks[j](xp, adapt=None, prompt=None, rank_prompt=None, block_weight=None)
            shared_prefix_x = xp

        if self.config.ffn_adapt:
            for i in range(len(self.adapter_list)):
                if shared_prefix_x is not None:
                    x = shared_prefix_x
                else:
                    x = copy.deepcopy(x_init)
                for j in range(prefix_boundary, len(self.blocks)):

                    rank_prompt = None
                    prompt = None

                    if j in self.adapt_pos:
                        if j in self.general_pos:
                            adapt = self._get_general_adapter_for_snapshot(i, j)
                            direction_scale = None
                        else:
                            pos = self.specific_pos.index(j)
                            adapt = self.adapter_list[i][pos]
                            direction_scale = self._inference_direction_scale(i, pos)

                        if self.use_block_weight and j in self.specific_pos:
                            pos_spec = self.specific_pos.index(j)
                            block_weight = self._get_block_weight_column(self.block_weight_list[i], pos_spec)
                        else:
                            block_weight = None
                        x = self.blocks[j](x, adapt, prompt, rank_prompt, block_weight, direction_scale)

                    else:
                        x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)

                x = self.norm(x)
                features.append(x)

            if shared_prefix_x is not None:
                x = shared_prefix_x
            else:
                x = copy.deepcopy(x_init)
            for i in range(prefix_boundary, len(self.blocks)):

                rank_prompt = None
                prompt = None

                if i in self.adapt_pos:
                    pos = self.adapt_pos.index(i)
                    adapt = self.cur_adapter[pos]
                    if i in self.specific_pos:
                        pos_spec = self.specific_pos.index(i)
                        if self.use_block_weight:
                            block_weight = self._get_block_weight_column(self.block_weight, pos_spec)
                        else:
                            block_weight = None
                        direction_scale = self._select_direction_scale_for_block(self.direction_scale, pos_spec)
                    else:
                        block_weight = None
                        direction_scale = None
                    x = self.blocks[i](x, adapt, prompt, rank_prompt, block_weight, direction_scale)
                else:
                    x = self.blocks[i](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            x = self.norm(x)
            features.append(x)

        return features

    def forward(self, x, test=False):
        if not test:
            output = self.forward_train(x)
            return output

        else:
            features = self.forward_test(x)
            output = torch.Tensor().to(features[0].device)
            for x in features:
                cls = x[:, 0, :]
                output = torch.cat((
                    output,
                    cls
                ), dim=1)
            return output

    def forward_proto(self, x, adapt_index):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)

        # the init_PTM's feature
        if adapt_index == -1:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            output = x[:, 0, :]
            return output

        i = adapt_index
        x = copy.deepcopy(x_init)
        if self.config.ffn_adapt:
            if i < len(self.adapter_list):
                for j in range(len(self.blocks)):

                    rank_prompt = None
                    prompt = None

                    if j in self.adapt_pos:
                        if j in self.general_pos:
                            adapt = self._get_general_adapter_for_snapshot(i, j)
                            direction_scale = None
                        else:
                            pos = self.specific_pos.index(j)
                            adapt = self.adapter_list[i][pos]
                            direction_scale = self._inference_direction_scale(i, pos)
                        if self.use_block_weight and j in self.specific_pos:
                            pos_spec = self.specific_pos.index(j)
                            block_weight = self._get_block_weight_column(self.block_weight_list[i], pos_spec)
                        else:
                            block_weight = None
                        x = self.blocks[j](x, adapt, prompt, rank_prompt, block_weight, direction_scale)

                    else:
                        x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            else:
                for j in range(len(self.blocks)):
                    rank_prompt = None
                    prompt = None

                    if j in self.adapt_pos:
                        pos = self.adapt_pos.index(j)
                        adapt = self.cur_adapter[pos]
                        if j in self.specific_pos:
                            pos_spec = self.specific_pos.index(j)
                            if self.use_block_weight:
                                block_weight = self._get_block_weight_column(self.block_weight, pos_spec)
                            else:
                                block_weight = None
                            direction_scale = self._select_direction_scale_for_block(self.direction_scale, pos_spec)
                        else:
                            block_weight = None
                            direction_scale = None

                        x = self.blocks[j](x, adapt, prompt, rank_prompt, block_weight, direction_scale)
                    else:
                        x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
        else:
            for j in range(len(self.blocks)):
                rank_prompt = None
                prompt = None

                x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)

        x = self.norm(x)
        output = x[:, 0, :]

        return output

    def forward_general_cls(self, x, t_idx):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_teacher = copy.deepcopy(x)

        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.cur_adapter[pos]
            x = self.blocks[j](x, adapt)

        x = self.norm(x)
        output_new = x[:, 0, :]



        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.old_adapter_list[t_idx-1][pos]
            x_teacher = self.blocks[j](x_teacher, adapt)
        x_teacher = self.norm(x_teacher)
        output_teacher= x_teacher[:, 0, :]

        return output_new, output_teacher



def vit_base_patch16_224_cllora(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint_model=timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if not model.msa_adapt:
        for adapter_temp in model.cur_adapter:
            for param in adapter_temp.lora_B.parameters():
                param.requires_grad = False
    else:
        for i in model.adapt_pos:
            if i in model.general_pos:
                pos = model.adapt_pos.index(i)
                for j in range(len(model.msa)):
                    if model.msa[j] == 1:
                        for param in model.cur_adapter[pos][j].lora_B.parameters():
                            param.requires_grad = False

    model._apply_alpha_mode_requires_grad()
    model._set_specific_trainable_state()
    return model

def vit_base_patch16_224_in21k_cllora(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    checkpoint_model=timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False


    if not model.msa_adapt:
        for adapter_temp in model.cur_adapter:
            for param in adapter_temp.lora_B.parameters():
                param.requires_grad = False
    else:
        for i in model.adapt_pos:
            if i in model.general_pos:
                pos = model.adapt_pos.index(i)
                for j in range(len(model.msa)):
                    if model.msa[j] == 1:
                        for param in model.cur_adapter[pos][j].lora_B.parameters():
                            param.requires_grad = False

    model._apply_alpha_mode_requires_grad()
    model._set_specific_trainable_state()
    return model


def load_npz_to_state_dict(filename):
    # Load the .npz file
    with np.load(filename, allow_pickle=True) as data:
        state_dict = {}
        for key in data.keys():
            state_dict[key] = torch.from_numpy(data[key])
    return state_dict

def compute_column_importance(matrix):
    """
    Compute importance of each column based on SVD and scale to range (0, 1).
    """
    U, S, Vt = torch.linalg.svd(matrix.T, full_matrices=False)
    importance_scores = torch.sum(torch.abs(U * S), dim=1)
    scaled_scores = (importance_scores - torch.min(importance_scores)) / (torch.max(importance_scores) - torch.min(importance_scores))
    epsilon = 1e-10
    scaled_scores = torch.maximum(scaled_scores, torch.tensor(epsilon))
    return scaled_scores


def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))
