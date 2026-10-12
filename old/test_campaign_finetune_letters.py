import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import campaign_finetune_letters as campaign
from utils.dataloader import InkVolumeDataset, MultiScrollIterableDataset
from utils.model import create_model
from utils.training_utils import create_optimizer_and_scheduler
from utils.visualizer import TensorboardVisualizer, _configured_surface_window_offset, predict_tiles


def config_for(tid):
    config = campaign.build_config(campaign._test(tid, campaign.DEFAULT_INIT_WEIGHTS))
    config.device = "cpu"
    return config


class LetterCampaignTest(unittest.TestCase):
    def test_six_patch_mae_recipe(self):
        import mae_pretrain_letters as run

        command = run.training_command(dry_run=True)
        self.assertEqual(len({patch[0] for patch in run.PATCHES}), 6)
        self.assertIn("auto_grown_20260927143929299_abf", {patch[1] for patch in run.PATCHES})
        self.assertIn("auto_grown_20260927154033403_abf", {patch[1] for patch in run.PATCHES})
        self.assertIn(str(campaign.FLAT_FORWARD[3]), command)
        self.assertNotIn(str(campaign.FLAT_IDS[campaign.OFFSET_ID]), command)
        for flag in ("--early-2d-unet", "--gated-stems", "--fiber-coordinate-branch", "--require-all-scrolls"):
            self.assertIn(flag, command)
        for flag, value in (("--ctx", "96"), ("--ds", "1"), ("--depth", "8"),
                            ("--d-start", "9"), ("--d-end", "17"), ("--data-norm-mode", "surface_anchor")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--freeze-loaded-backbone", command)
        self.assertNotEqual(str(run.INIT_WEIGHTS), campaign.DEFAULT_INIT_WEIGHTS)

    def test_focused_queue(self):
        self.assertEqual(len(campaign.FOCUS_OVERRIDES), 7)
        for tid in campaign.FOCUS_OVERRIDES:
            config = config_for(tid)
            self.assertEqual(config.data.vis_scroll_ids, [20261012073246, 20261012065806])
            self.assertEqual(config.tra.eval_int_scrolls, 2)
            self.assertEqual(config.tra.n_epochs, 6)
            self.assertEqual(config.tra.pu_lambda, 0)
            self.assertEqual(config.data.far_negative_share, 0)
            self.assertEqual(config.data.depth_jitter, 0)
            self.assertEqual(len(config.scroll_ids()), 5)
            if not tid.startswith("focus_encoder"):
                self.assertEqual(config.tra.freeze_prefixes, campaign.FREEZE_PREFIXES)
        self.assertEqual(config_for("focus_ratio1").data.train_scroll_weights, [1, 1])
        for tid, offset in (("focus_depth_m1", -1), ("focus_depth_p1", 1)):
            config = config_for(tid)
            self.assertEqual(config.data.surface_window_offset_by_scroll,
                             {sid: offset for sid in campaign.FLAT_OFFSETEXTRA})

    def test_refinement_defaults(self):
        self.assertEqual(campaign.DEFAULT_ARMS, ("refine_baseline_m1", "refine_depth_m2", "refine_encoder_3e6"))
        for tid in campaign.DEFAULT_ARMS:
            config = config_for(tid)
            self.assertEqual((config.data.ring_close_r, config.data.ring_gap_r, config.data.ring_shell_r), (2, 2, 4))
            self.assertTrue(config.data.multitile_pos_only)
            self.assertEqual(config.tra.pu_lambda, 0)
            self.assertEqual(config.data.far_negative_share, 0)
            self.assertEqual(config.data.vis_scroll_ids, campaign.FOCUS_VIS_IDS)
            self.assertEqual(config.data.train_scroll_weights, [5, 1])
            offset = -2 if tid == "refine_depth_m2" else -1
            self.assertEqual(config.data.surface_window_offset_by_scroll,
                             {sid: offset for sid in campaign.FLAT_OFFSETEXTRA})
            self.assertEqual(config.tra.n_epochs, 6)

    def test_slow_encoder_rates_survive_epoch_update(self):
        from train import Trainer

        for tid, expected in (("focus_encoder_1e6", 1e-6), ("focus_encoder_1e5", 1e-5),
                      ("refine_encoder_3e6", 3e-6)):
            config = config_for(tid)
            model, _ = create_model(config)
            optimizer, scheduler = create_optimizer_and_scheduler(model, config)
            trainer = Trainer.__new__(Trainer)
            trainer.c, trainer.optimizer = config, optimizer
            for epoch in (0, 1, 5):
                trainer._update_encoder_optimization(epoch)
                rates = {id(parameter): group["lr"] for group in optimizer.param_groups for parameter in group["params"]}
                for name, parameter in model.named_parameters():
                    rate = expected if name.startswith(campaign.FREEZE_PREFIXES) else 1e-4
                    self.assertAlmostEqual(rates[id(parameter)], rate, places=12, msg=(tid, name))
                    self.assertTrue(parameter.requires_grad)
                scheduler.step(1.0)

    def test_all_arms_preserve_sparse_geometry_and_holdout(self):
        for tid in campaign.ARMS:
            with self.subTest(tid=tid):
                config = config_for(tid)
                trained = {sid for ids in config.data.train_scroll_dict.values() for sid in ids}
                self.assertEqual(trained, set(config.scroll_ids()))
                self.assertTrue(trained.isdisjoint(config.data.label_free_vis_scroll_ids))
                self.assertEqual(config.data.label_free_vis_scroll_ids, [campaign.FLAT_FORWARD[3]])
                self.assertEqual(config.data.context_size, 96)
                self.assertEqual(config.data.depth, 8)
                self.assertEqual(config.model.multitile_grid * config.model.multitile_subtile, 64)
                self.assertTrue(config.data.multitile_ring_gate)
                self.assertEqual(config.data.edge_soft_sigma, 8.0)
                self.assertFalse(config.dl.data_aug)
                self.assertEqual(config.tra.eval_int, config.tra.n_epochs)
                self.assertEqual(config.init_weights, campaign.DEFAULT_INIT_WEIGHTS)
                self.assertEqual(len(config.data.train_scroll_weights), len(config.data.train_scroll_dict))
                if config.tra.pu_lambda == 0:
                    self.assertEqual(config.data.far_negative_share, 0)

    def test_overrides_and_control_groups(self):
        for tid, overrides in campaign.DIAGNOSTIC_OVERRIDES.items():
            config = config_for(tid)
            for key, value in overrides.items():
                section, field = key.split(".")
                self.assertEqual(getattr(getattr(config, section), field), value, (tid, key))
        self.assertEqual(config_for("diag_weight20").data.train_scroll_weights, [20, 1])
        self.assertNotIn(campaign.PHERC0009B_ID, config_for("diag_0846_only").scroll_ids())
        self.assertIn(campaign.PHERC0009B_ID, config_for("diag_0846_only").data.vis_scroll_ids)
        for tid in ("diag_long", "diag_unfreeze_long"):
            config = config_for(tid)
            self.assertEqual(config.tra.n_epochs, 18)
            self.assertGreaterEqual(config.tra.patience, config.tra.n_epochs)
            for tid in ("diag_no_bag", "diag_weight20", "diag_0846_only", "diag_depth_m2",
                    "diag_depth_p2", "diag_tiles8", "diag_loose_ring", "diag_unfreeze_long"):
                config = config_for(tid)
                self.assertEqual(config.tra.freeze_prefixes, ())
                self.assertEqual(config.tra.pu_lambda, 1.0)
                self.assertEqual(config.data.far_negative_share, 0.15)

    def test_optimizer_freeze_and_sparse_head_compatibility(self):
        torch.set_num_threads(2)
        base = config_for(campaign.DIAGNOSTIC_BASE)
        model, _ = create_model(base)
        state = copy.deepcopy(model.state_dict())
        for tid in (campaign.DIAGNOSTIC_BASE, "diag_unfreeze", "diag_stem", "diag_tiles8"):
            config = config_for(tid)
            model, _ = create_model(config)
            model.load_state_dict(state, strict=True)
            optimizer, _ = create_optimizer_and_scheduler(model, config)
            optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            for name, parameter in model.named_parameters():
                frozen = name.startswith(tuple(config.tra.freeze_prefixes))
                self.assertEqual(parameter.requires_grad, not frozen, (tid, name))
                self.assertEqual(id(parameter) in optimized, not frozen, (tid, name))

    def test_sampling_is_domain_weighted_not_letter_count_weighted(self):
        class Child:
            def __init__(self, name, length):
                self.name, self.length = name, length

            def __len__(self):
                return self.length

            def __iter__(self):
                return iter([self.name] * self.length)

        dataset = MultiScrollIterableDataset(
            [Child(index, 6 if index < 4 else 576) for index in range(5)],
            balance_scrolls=True, sampling_groups=[[0, 1, 2, 3], [4]], sampling_weights=[5, 1],
        )
        self.assertEqual(Counter(dataset), Counter({0: 125, 1: 125, 2: 125, 3: 125, 4: 100}))

    def test_manifest_keeps_dynamic_fields(self):
        test = campaign._test("diag_unfreeze_no_pu", campaign.DEFAULT_INIT_WEIGHTS)
        config = campaign.build_config(test)
        with tempfile.TemporaryDirectory() as directory, patch.object(campaign, "ROOT", Path(directory)):
            campaign._write_diagnostic_manifest(test, config)
            payload = json.loads((Path(directory) / config.model_dir / "letters_experiment.json").read_text())
        self.assertEqual(payload["tra"]["freeze_prefixes"], [])
        self.assertEqual(payload["data"]["train_only_scroll_ids"], campaign.FLAT_OFFSETEXTRA)
        self.assertEqual(payload["data"]["train_scroll_weights"], [5, 1])

    def test_figure_callers_forward_the_configured_offset(self):
        config = config_for("diag_depth_m2")
        sid = campaign.FLAT_FORWARD[0]
        mask = np.ones((128, 128), np.uint8)
        volume = np.zeros((28, 128, 128), np.float32)
        visualizer = SimpleNamespace(
            c=config, scroll1_id=sid, mask=mask, manual_split=True, manual_train_mask=mask,
            full_y_range=(0, 128), full_x_range=(0, 128), volume=volume,
            global_mean=0, global_std=1, global_min=0, global_max=1,
            surface_depth_map=mask * 13, surface_confidence_map=mask * 255,
            _hard_mining_dir=lambda: "unused",
            _gen_tile_coords=lambda *args, **kwargs: [(0, 56, 56)],
            _frag_bbox=lambda *args, **kwargs: (0, 128, 0, 128),
        )
        fragment = {"name": "probe", "id": sid, "volume": volume, "mask": mask,
                    "mean": 0, "std": 1, "min": 0, "max": 1,
                    "surface_depth": mask * 13, "surface_confidence": mask * 255}
        for render, arguments in (
            (TensorboardVisualizer.add_evaluation_figures, (0, torch.nn.Identity())),
            (TensorboardVisualizer._render_test_fragment, (0, torch.nn.Identity(), fragment)),
        ):
            with patch("utils.visualizer.predict_tiles", side_effect=RuntimeError("capture input")) as predictor:
                with self.assertRaisesRegex(RuntimeError, "capture input"):
                    render(visualizer, *arguments)
                self.assertEqual(predictor.call_args.kwargs["surface_depth_offset"], -2)

    def test_pure_scroll_arm_handles_empty_validation(self):
        from train import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.c = config_for("diag_0846_only")
        trainer.model = torch.nn.Identity()
        trainer.valid_loader = []
        trainer._divdis_loader = None
        metrics = trainer.validate_epoch()
        self.assertEqual(metrics["loss"], 0.0)
        self.assertEqual(metrics["scores"], [])

    def test_train_and_figure_read_same_depth_window(self):
        class Capture(torch.nn.Module):
            def forward(self, images, **kwargs):
                self.images = images.clone()
                self.depth = kwargs["teacher_surface_depth"].clone()
                return images.new_zeros((len(images), 16))

        surface = np.full((128, 128), 13, np.uint8)
        confidence = np.full((128, 128), 255, np.uint8)
        volume = np.broadcast_to(np.arange(28, dtype=np.float32)[:, None, None], (28, 128, 128))
        for tid, expected in (("diag_depth_m2", 8), ("diag_unfreeze_no_pu", 10), ("diag_depth_p2", 12),
                      ("focus_depth_m1", 9), ("focus_depth_p1", 11)):
            config = config_for(tid)
            config.data.eval_prefetch = 0
            sid = campaign.FLAT_FORWARD[0]
            offset = _configured_surface_window_offset(config, sid)
            fake = SimpleNamespace(surface_depth=surface, surface_confidence=confidence,
                                   _mt=True, _mt_grid=4, _mt_sub=16, tile_size=16, depth=8,
                                   z_start=4, _surface_window_offset=offset)
            self.assertEqual(InkVolumeDataset._surface_centered_start(fake, 16, 16, 96, 28, 56, 56), expected)
            model = Capture()
            predict_tiles(config, model, volume, np.ones((128, 128)), [(0, 56, 56)],
                          (0, 128), (0, 128), 4, "probe", 0, 1, 0, 27,
                          surface_depth_map=surface, surface_confidence_map=confidence,
                          surface_depth_offset=offset)
            self.assertAlmostEqual(float(model.images[0, 0, 0, 0, 0]) * 27, expected, places=5)
            self.assertEqual(float(model.depth.flatten()[0]), 13 - expected)
            self.assertEqual(_configured_surface_window_offset(config, campaign.PHERC0009B_ID), 0)
            config.data.surface_window_offset_by_scroll = {str(sid): offset}
            self.assertEqual(_configured_surface_window_offset(config, sid), offset)


if __name__ == "__main__":
    unittest.main()