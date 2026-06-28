import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from backbone.vit_cllora import Adapter_lora, SDLoRAAdapter, VisionTransformer
from models.cllora import compute_optional_orthogonality_loss
from utils.inc_net import get_backbone, parse_cllora_adapter_options


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
        "ffn_num": 4,
        "d_model": 8,
        "vpt_on": False,
        "vpt_num": 0,
        "_device": "cpu",
        "direction_scale_init": 0.05,
        "direction_norm_eps": 1e-6,
        "block_weight_normalization": "mean_l1",
        "specific_lora_init_scale": 1e-3,
        # SD-LoRA-RR task-specific adapter settings (flat attrs, as inc_net produces)
        "sd_lora_enable": True,
        "sd_lora_variant": "rr",
        "sd_lora_r0": 4,
        "sd_lora_r_min": 2,
        "sd_lora_rank_decay": 0.8,
        "normalize_direction": True,
        "alpha_mode": "task_conditioned",
        "train_old_alpha": False,
        "cache_old_directions": True,
        "cache_device": "cuda",
        "combine_directions_before_linear": False,
        "specific_lora_rank_schedule": None,
        "nb_tasks": 3,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def make_model(**overrides):
    torch.manual_seed(7)
    return VisionTransformer(
        img_size=8,
        patch_size=4,
        in_chans=3,
        num_classes=0,
        embed_dim=8,
        depth=4,
        num_heads=2,
        mlp_ratio=2,
        qkv_bias=True,
        tuning_config=make_config(**overrides),
    )


def first_specific_sd_adapter(model):
    pos = model.adapt_pos.index(model.specfic_pos[0])
    return model.cur_adapter[pos][0]


def freeze_base_weights_like_pretrained_cllora(model):
    for name, param in model.named_parameters():
        if (
            "cur_adapter" not in name
            and "block_weight" not in name
            and "direction_scale" not in name
        ):
            param.requires_grad = False


class SDLoRARRTests(unittest.TestCase):
    def test_optional_orthogonality_loss_zero_when_disabled(self):
        previous = [torch.randn(3, 2)]
        current = torch.randn(3, 2, requires_grad=True)

        loss = compute_optional_orthogonality_loss(
            previous,
            current,
            use_orthogonal_constraint=False,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(loss.requires_grad)

    def test_optional_orthogonality_loss_zero_without_block_weight(self):
        loss = compute_optional_orthogonality_loss(
            previous_weights_list=[],
            current_weights=None,
            use_orthogonal_constraint=True,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(loss.requires_grad)

    def test_optional_orthogonality_loss_enabled_matches_original_logic(self):
        previous = [torch.tensor([[1.0, 0.0], [0.0, 0.0]])]
        current = torch.tensor([[1.0, 1.0], [0.0, 0.0]], requires_grad=True)

        loss = compute_optional_orthogonality_loss(
            previous,
            current,
            use_orthogonal_constraint=True,
        )

        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(current.grad)

    def test_block_weight_trains_and_affects_specific_forward_without_orthogonal_loss(self):
        model = make_model(use_orthogonal_constraint=False)
        self.assertTrue(model.block_weight.requires_grad)

        inputs = torch.randn(2, 3, 8, 8)
        output = model.forward_train(inputs)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()

        self.assertIsNotNone(model.block_weight.grad)
        self.assertGreater(model.block_weight.grad.abs().sum().item(), 0.0)

        with torch.no_grad():
            baseline = model.forward_train(inputs)
            model.block_weight[0, 0] += 1.0
            changed = model.forward_train(inputs)
        self.assertGreater((baseline - changed).abs().max().item(), 0.0)

    def test_shared_positions_keep_original_adapter_class(self):
        model = make_model()
        for block_idx in model.general_pos:
            pos = model.adapt_pos.index(block_idx)
            self.assertIsInstance(model.cur_adapter[pos][0], Adapter_lora)
            self.assertNotIsInstance(model.cur_adapter[pos][0], SDLoRAAdapter)

        for block_idx in model.specfic_pos:
            pos = model.adapt_pos.index(block_idx)
            self.assertIsInstance(model.cur_adapter[pos][0], SDLoRAAdapter)

    def test_specific_sd_lora_forward_works_without_block_weight(self):
        model = make_model(use_block_weight=False)
        output = model.forward_train(torch.randn(2, 3, 8, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_historical_direction_frozen_current_direction_and_scales_train(self):
        model = make_model()
        model.add_adapter_to_list()

        adapter = first_specific_sd_adapter(model)
        # rr schedule: r0=4, decay=0.8 -> task0=4, task1=floor(4*0.8)=3
        self.assertEqual(adapter.direction_ranks(), [4, 3])
        old_weight_before = adapter.directions[0].lora_A.weight.detach().clone()
        current_weight_before = adapter.directions[1].lora_A.weight.detach().clone()

        self.assertFalse(adapter.directions[0].lora_A.weight.requires_grad)
        self.assertFalse(adapter.directions[0].lora_B.weight.requires_grad)
        self.assertTrue(adapter.directions[1].lora_A.weight.requires_grad)
        self.assertTrue(adapter.directions[1].lora_B.weight.requires_grad)
        self.assertTrue(model.direction_scale.requires_grad)
        self.assertEqual(tuple(model.direction_scale.shape), (len(model.specfic_pos), 2))

        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.1,
        )
        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertNotIn(id(adapter.directions[0].lora_A.weight), optimizer_param_ids)
        self.assertIn(id(adapter.directions[1].lora_A.weight), optimizer_param_ids)
        self.assertIn(id(model.direction_scale), optimizer_param_ids)

        output = model.forward_train(torch.randn(2, 3, 8, 8))
        loss = output.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()

        self.assertIsNone(adapter.directions[0].lora_A.weight.grad)
        self.assertIsNotNone(adapter.directions[1].lora_A.weight.grad)
        self.assertIsNotNone(model.direction_scale.grad)
        self.assertGreater(model.direction_scale.grad[:, 0].abs().sum().item(), 0.0)

        optimizer.step()

        self.assertTrue(torch.equal(old_weight_before, adapter.directions[0].lora_A.weight))
        self.assertGreater((current_weight_before - adapter.directions[1].lora_A.weight).abs().sum().item(), 0.0)

    def test_snapshot_forward_is_stable_after_later_task_update(self):
        model = make_model()
        freeze_base_weights_like_pretrained_cllora(model)
        inputs = torch.randn(2, 3, 8, 8)
        model.add_adapter_to_list()

        with torch.no_grad():
            before = model.forward_proto(inputs, adapt_index=0)

        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.1,
        )
        output = model.forward_train(inputs)
        loss = output.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            after = model.forward_proto(inputs, adapt_index=0)

        self.assertTrue(torch.allclose(before, after, atol=0.0, rtol=0.0))

    def test_state_dict_round_trip_preserves_snapshot_output(self):
        model = make_model()
        inputs = torch.randn(2, 3, 8, 8)
        model.add_adapter_to_list()
        model.add_adapter_to_list()

        clone = copy.deepcopy(model)
        clone.load_state_dict(model.state_dict())

        with torch.no_grad():
            original = model.forward_proto(inputs, adapt_index=0)
            restored = clone.forward_proto(inputs, adapt_index=0)

        self.assertTrue(torch.allclose(original, restored, atol=0.0, rtol=0.0))

    def test_fresh_model_load_rebuilds_sd_lora_rr_state(self):
        model = make_model()
        inputs = torch.randn(2, 3, 8, 8)
        model.add_adapter_to_list()
        model.add_adapter_to_list()

        restored = make_model()
        restored.load_state_dict(model.state_dict())
        self.assertIn("specific_lora_rank_history_buffer", model.state_dict())

        with torch.no_grad():
            original = model.forward_proto(inputs, adapt_index=1)
            loaded = restored.forward_proto(inputs, adapt_index=1)

        self.assertTrue(torch.allclose(original, loaded, atol=0.0, rtol=0.0))
        self.assertEqual(len(restored.adapter_list), 2)
        self.assertEqual(len(restored.block_weight_list), 2)
        self.assertEqual(len(restored.direction_scale_list), 2)
        self.assertTrue(restored.block_weight.requires_grad)
        self.assertTrue(restored.direction_scale.requires_grad)
        self.assertFalse(restored.block_weight_list[0].requires_grad)
        self.assertFalse(restored.direction_scale_list[0].requires_grad)

        adapter = first_specific_sd_adapter(restored)
        # rr schedule: r0=4, decay=0.8 -> [4, floor(3.2)=3, floor(2.56)=2]
        self.assertEqual(adapter.direction_ranks(), [4, 3, 2])
        self.assertEqual(restored.specific_lora_rank_history_buffer.tolist(), [4, 3, 2])
        self.assertFalse(adapter.directions[0].lora_A.weight.requires_grad)
        self.assertTrue(adapter.directions[-1].lora_A.weight.requires_grad)

    def test_rr_lowers_new_direction_params_and_outputs_expected_feature_width(self):
        model = make_model()
        first_adapter = first_specific_sd_adapter(model)
        first_direction_params = sum(p.numel() for p in first_adapter.directions[0].parameters())

        model.add_adapter_to_list()
        second_adapter = first_specific_sd_adapter(model)
        second_direction_params = sum(p.numel() for p in second_adapter.directions[1].parameters())
        self.assertLess(second_direction_params, first_direction_params)

        model.add_adapter_to_list()
        with torch.no_grad():
            output = model.forward(torch.randn(2, 3, 8, 8), test=True)

        self.assertEqual(output.shape, (2, 3 * model.embed_dim))
        self.assertTrue(torch.isfinite(output).all())

    def test_rr_rank_formula(self):
        # r_t = max(r_min, floor(r0 * decay^(t-1))), task_id zero-based.
        model = make_model(sd_lora_r0=8, sd_lora_r_min=2, sd_lora_rank_decay=0.8)
        self.assertEqual(model.get_specific_lora_rank(0), 8)   # floor(8*0.8^0)=8
        self.assertEqual(model.get_specific_lora_rank(1), 6)   # floor(8*0.8)=6
        self.assertEqual(model.get_specific_lora_rank(2), 5)   # floor(8*0.64)=5
        self.assertEqual(model.get_specific_lora_rank(10), 2)  # clamped to r_min

    def test_fixed_variant_keeps_rank(self):
        model = make_model(sd_lora_variant="fixed", sd_lora_r0=8)
        self.assertEqual(model.get_specific_lora_rank(0), 8)
        self.assertEqual(model.get_specific_lora_rank(5), 8)

    def test_enable_false_keeps_single_direction_per_task(self):
        model = make_model(sd_lora_enable=False, sd_lora_r0=4, normalize_direction=False)
        adapter = first_specific_sd_adapter(model)
        self.assertIsInstance(adapter, Adapter_lora)
        self.assertNotIsInstance(adapter, SDLoRAAdapter)
        self.assertFalse(model.direction_scale.requires_grad)

        model.add_adapter_to_list()  # next task
        adapter = first_specific_sd_adapter(model)
        self.assertIsInstance(adapter, Adapter_lora)
        self.assertNotIsInstance(adapter, SDLoRAAdapter)
        self.assertFalse(model.direction_scale.requires_grad)

    def test_direction_scale_is_block_specific(self):
        model = make_model()
        model.add_adapter_to_list()

        self.assertEqual(tuple(model.direction_scale.shape), (len(model.specfic_pos), 2))
        self.assertEqual(tuple(model.direction_scale_list[0].shape), (len(model.specfic_pos), 1))
        self.assertEqual(model._inference_direction_scale(0, 0).shape[0], 1)
        self.assertEqual(model._inference_direction_scale(0, 1).shape[0], 1)

    def test_normalize_direction_off_returns_raw_weight(self):
        model_on = make_model(normalize_direction=True)
        adapter_on = first_specific_sd_adapter(model_on)
        d = adapter_on.directions[0]
        raw = d.lora_A.weight @ d.lora_B.weight
        normalized = adapter_on._normalized_direction_weight(d)
        # when norm != 0, normalized differs from raw unless raw is already unit-norm
        self.assertFalse(torch.allclose(normalized, raw))

        model_off = make_model(normalize_direction=False)
        adapter_off = first_specific_sd_adapter(model_off)
        d2 = adapter_off.directions[0]
        raw2 = d2.lora_A.weight @ d2.lora_B.weight
        self.assertTrue(torch.allclose(adapter_off._normalized_direction_weight(d2), raw2))

    def test_parse_sd_lora_cache_defaults_and_overrides(self):
        defaults = parse_cllora_adapter_options({"sd_lora": {"enable": True}}, ffn_num=8)
        self.assertTrue(defaults["cache_old_directions"])
        self.assertEqual(defaults["cache_device"], "cuda")
        self.assertFalse(defaults["combine_directions_before_linear"])

        overrides = parse_cllora_adapter_options(
            {
                "sd_lora": {
                    "enable": True,
                    "cache_old_directions": False,
                    "cache_device": "cpu",
                    "combine_directions_before_linear": True,
                }
            },
            ffn_num=8,
        )
        self.assertFalse(overrides["cache_old_directions"])
        self.assertEqual(overrides["cache_device"], "cpu")
        self.assertTrue(overrides["combine_directions_before_linear"])

    def test_get_backbone_passes_sd_lora_cache_options_to_tuning_config(self):
        class FakeModel:
            def eval(self):
                return self

        args = {
            "backbone_type": "vit_base_patch16_224_in21k_cllora",
            "model_name": "cllora",
            "device": ["cpu"],
            "ffn_num": 8,
            "use_distillation": True,
            "msa_adapt": True,
            "msa": [1, 0, 1],
            "general_pos": [0, 1, 2, 3, 4, 5],
            "specfic_pos": [6, 7, 8, 9, 10, 11],
            "task_specific_adapter": {
                "use_block_weight": True,
                "use_orth_loss": False,
            },
            "sd_lora": {
                "enable": True,
                "cache_old_directions": False,
                "cache_device": "cpu",
                "combine_directions_before_linear": True,
            },
        }

        with patch("backbone.vit_cllora.vit_base_patch16_224_in21k_cllora", return_value=FakeModel()) as factory:
            model = get_backbone(args)

        tuning_config = factory.call_args.kwargs["tuning_config"]
        self.assertIsInstance(model, FakeModel)
        self.assertFalse(tuning_config.cache_old_directions)
        self.assertEqual(tuning_config.cache_device, "cpu")
        self.assertTrue(tuning_config.combine_directions_before_linear)

    def test_task_start_saves_old_direction_cache_without_caching_current_direction(self):
        model = make_model(cache_device="cpu")
        model.add_adapter_to_list()

        adapter = first_specific_sd_adapter(model)
        cached = adapter.cached_old_direction_weights

        self.assertEqual(adapter.num_cached_directions(), 1)
        self.assertFalse(adapter.current_direction_cached())
        self.assertEqual(tuple(cached.shape), (1, model.embed_dim, model.embed_dim))
        self.assertEqual(cached.device.type, "cpu")
        self.assertFalse(cached.requires_grad)
        expected = adapter._normalized_direction_weight(adapter.directions[0]).detach().cpu()
        self.assertTrue(torch.allclose(cached[0], expected))

    def test_cached_old_direction_forward_does_not_recompute_old_direction(self):
        model = make_model(cache_device="cpu")
        model.add_adapter_to_list()
        adapter = first_specific_sd_adapter(model)
        x = torch.randn(2, 3, model.embed_dim)
        direction_scale = torch.tensor([1.0, 0.0])

        with torch.no_grad():
            before = adapter(x, direction_scale)
            adapter.directions[0].lora_A.weight.add_(100.0)
            after = adapter(x, direction_scale)

        self.assertTrue(torch.allclose(before, after, atol=0.0, rtol=0.0))

    def test_missing_old_direction_cache_warns_and_falls_back(self):
        model = make_model(cache_device="cpu")
        model.add_adapter_to_list()
        adapter = first_specific_sd_adapter(model)
        adapter.clear_cached_old_directions()

        with self.assertLogs(level="WARNING") as logs:
            out = adapter(torch.randn(2, 3, model.embed_dim), torch.tensor([1.0, 0.0]))

        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(any("SD-LoRA old direction cache missing" in line for line in logs.output))

    def test_current_direction_stays_dynamic_and_trainable_with_old_cache(self):
        model = make_model(cache_device="cpu")
        model.add_adapter_to_list()
        adapter = first_specific_sd_adapter(model)
        x = torch.randn(2, 3, model.embed_dim)
        direction_scale = torch.tensor([0.0, 1.0])

        self.assertTrue(adapter.directions[-1].lora_A.weight.requires_grad)
        self.assertFalse(adapter.cached_old_direction_weights.requires_grad)

        with torch.no_grad():
            before = adapter(x, direction_scale)
            adapter.directions[-1].lora_A.weight.add_(0.5)
            after = adapter(x, direction_scale)

        self.assertGreater((before - after).abs().max().item(), 0.0)

    def test_combine_directions_before_linear_matches_separate_linear(self):
        model = make_model(cache_device="cpu")
        model.add_adapter_to_list()
        adapter = first_specific_sd_adapter(model)
        x = torch.randn(2, 3, model.embed_dim)
        direction_scale = torch.tensor([0.7, -0.2])

        adapter.combine_directions_before_linear = False
        with patch("backbone.vit_cllora.F.linear", wraps=F.linear) as separate_linear:
            separate = adapter(x, direction_scale)
        self.assertEqual(separate_linear.call_count, 2)

        adapter.combine_directions_before_linear = True
        with patch("backbone.vit_cllora.F.linear", wraps=F.linear) as combined_linear:
            combined = adapter(x, direction_scale)
        self.assertEqual(combined_linear.call_count, 1)
        self.assertTrue(torch.allclose(separate, combined, atol=1e-6, rtol=1e-6))

    def test_task_conditioned_freezes_old_alpha_routing(self):
        model = make_model(alpha_mode="task_conditioned", train_old_alpha=False)
        model.add_adapter_to_list()
        # current routing trainable, old snapshot frozen
        self.assertTrue(model.direction_scale.requires_grad)
        self.assertEqual(len(model.direction_scale_list), 1)
        self.assertFalse(model.direction_scale_list[0].requires_grad)
        # inference branch 0 uses its own saved snapshot
        self.assertTrue(torch.equal(model._inference_direction_scale(0, 0), model.direction_scale_list[0][0]))

    def test_global_alpha_inference_uses_latest_prefix(self):
        model = make_model(alpha_mode="global")
        model.add_adapter_to_list()
        # global: branch 0 reuses the prefix of the current shared routing vector
        ref = model._inference_direction_scale(0, 0)
        self.assertEqual(ref.shape[0], 1)
        self.assertTrue(torch.equal(ref, model.direction_scale[0, :1]))

    def test_train_old_alpha_unfreezes_snapshots(self):
        model = make_model(alpha_mode="task_conditioned", train_old_alpha=True)
        model.add_adapter_to_list()
        self.assertTrue(model.direction_scale_list[0].requires_grad)

    def test_legacy_config_without_sd_lora_block_disables_sd_lora(self):
        options = parse_cllora_adapter_options(
            {
                "specific_lora_rank_schedule": {
                    "milestones": [0.4, 0.8],
                    "ranks": [8, 6, 4],
                }
            },
            ffn_num=8,
        )

        self.assertFalse(options["sd_lora_enable"])
        self.assertEqual(options["sd_lora_variant"], "legacy")


if __name__ == "__main__":
    unittest.main()
