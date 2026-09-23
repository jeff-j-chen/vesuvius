import unittest

import torch

import campaign_archs_31 as campaign31
import campaign_archs_33 as campaign
from utils.model import create_model


def _config(tid):
    return campaign.build_config(next(test for test in campaign.TESTS if test["tid"] == tid))


class Campaign33Test(unittest.TestCase):
    def test_arm_matrix(self):
        self.assertEqual(
            [test["tid"] for test in campaign.TESTS],
            [
                "baseline",
                "early_patchdro_seed42",
                "mid_air_offset_m1",
                "lodo_pherc0172_mid",
                "lodo_pherc0172_early_patchdro",
                "lodo_phercparis4_mid",
                "lodo_phercparis4_early_patchdro",
                "early_patchdro_ema",
                "mid_groupdro_ema",
                "early_planar",
                "early_planar_patchdro",
                "early_patchdro_depth_interp",
                "early_patchdro_ink_band",
            ],
        )

    def test_paths_and_protocol(self):
        for test in campaign.TESTS:
            config = _config(test["tid"])
            self.assertEqual(config.tra.log_dir, campaign.LOG_DIR)
            self.assertTrue(config.model_dir.startswith(campaign.MODEL_DIR))
            self.assertEqual(config.exp_name, f"33_{test['tid']}")
            self.assertEqual(config.dl.batch_size, campaign31.FINETUNE_BATCH_SIZE)
            self.assertEqual(config.tra.lr, campaign31.FINETUNE_LR)
            self.assertEqual(config.data.depth, 8)
            self.assertFalse(config.tra.mae_anchor_lambda)
        self.assertEqual(campaign31.LOG_DIR, "./runs_archs31")
        self.assertEqual(campaign31.MODEL_DIR, "models/archs31")

    def test_arm_settings(self):
        self.assertEqual(_config("mid_air_offset_m1").data.surface_window_offset, -1)
        baseline = _config("baseline")
        self.assertTrue(baseline.model.mid_2d_unet and baseline.model.gated_stems)
        self.assertFalse(
            baseline.tra.physical_domain_groupdro
            or baseline.tra.physical_patch_groupdro
            or baseline.tra.model_ema
            or baseline.tra.pcgrad_gram
        )
        self.assertEqual(baseline.data.surface_window_offset, 0)
        self.assertEqual(baseline.data.surface_window_offset_by_scroll, {})
        self.assertEqual(baseline.tra.seed, 41)
        replicate = _config("early_patchdro_seed42")
        self.assertTrue(replicate.model.early_2d_unet and replicate.tra.physical_patch_groupdro)
        self.assertEqual(replicate.tra.seed, 42)
        ema = _config("early_patchdro_ema")
        self.assertTrue(ema.tra.model_ema and ema.tra.physical_patch_groupdro)
        mid = _config("mid_groupdro_ema")
        self.assertTrue(mid.tra.model_ema and mid.tra.physical_domain_groupdro)
        self.assertTrue(mid.model.mid_2d_unet)
        for domain in ("pherc0172", "phercparis4"):
            self.assertEqual(_config(f"lodo_{domain}_mid").data.holdout_domains, [domain])
            early = _config(f"lodo_{domain}_early_patchdro")
            self.assertEqual(early.data.holdout_domains, [domain])
            self.assertIn(domain, early.data.train_scroll_dict)
        planar = _config("early_planar_patchdro")
        self.assertTrue(planar.model.planar_early_convs and planar.tra.physical_patch_groupdro)
        self.assertIn("campaign33_early_gated_planar", planar.init_weights)

    def test_interp_depth_mask_never_zeroes(self):
        import random

        import numpy as np
        from utils.dataloader import Transform

        config = _config("early_patchdro_depth_interp")
        self.assertEqual((config.dl.depth_mask_prob, config.dl.depth_mask_mode), (0.3, "interp"))
        self.assertTrue(config.tra.physical_patch_groupdro and config.model.early_2d_unet)
        self.assertEqual(_config("early_patchdro_seed42").dl.depth_mask_prob, 0.0)
        transform = Transform(config)
        transform.depth_mask_prob = 1.0
        block = np.stack([np.full((4, 4), float((k + 1) ** 2)) for k in range(8)]).astype(np.float32)
        random.seed(0)
        for _ in range(20):
            out = transform._apply_depth_mask(block)
            changed = [k for k in range(8) if not np.allclose(out[k], block[k])]
            self.assertEqual(len(changed), 1)
            index = changed[0]
            neighbours = [k for k in (index - 1, index + 1) if 0 <= k < 8]
            self.assertTrue(np.allclose(out[index], block[neighbours].mean(axis=0)))
            self.assertGreater(float(out.min()), 0.0)

    def test_ink_band_offsets(self):
        from types import SimpleNamespace

        import numpy as np
        from utils.dataloader import InkVolumeDataset

        config = _config("early_patchdro_ink_band")
        offsets = config.data.surface_window_offset_by_scroll
        self.assertEqual(set(offsets), {int(s.scroll_id) for s in config.data.scrolls})
        self.assertTrue(all(-2 <= value <= 0 for value in offsets.values()))
        self.assertEqual(offsets[20260226000000], -2)
        self.assertEqual(offsets[20251111010954], 0)
        self.assertTrue(config.tra.physical_patch_groupdro and config.model.early_2d_unet)
        self.assertEqual(config.data.surface_window_offset, 0)

        fake = SimpleNamespace(
            surface_depth=np.full((64, 64), 12, dtype=np.uint8),
            surface_confidence=np.full((64, 64), 200, dtype=np.uint8),
            _mt=True, _mt_grid=4, _mt_sub=16, tile_size=16, depth=8, z_start=0,
            _surface_window_offset=offsets[20260226000000],
        )
        self.assertEqual(InkVolumeDataset._surface_centered_start(fake, 0, 0, 64, 28, 0, 0), 7)

    def test_planar_model_never_mixes_slices_before_collapse(self):
        config = _config("early_planar")
        config.device = "cpu"
        model, _ = create_model(config)
        for prefix in ("enc1.", "gated_cue_stem."):
            convs = [
                module for name, module in model.named_modules()
                if name.startswith(prefix) and isinstance(module, torch.nn.Conv3d)
            ]
            self.assertTrue(convs)
            self.assertTrue(all(conv.kernel_size[0] == 1 for conv in convs))
        model.eval()
        with torch.no_grad():
            out = model(
                torch.rand(1, 1, 8, 192, 192),
                teacher_surface_depth=torch.full((1, 192, 192), 3.5),
                teacher_surface_confidence=torch.ones(1, 192, 192),
            )
        self.assertEqual(tuple(out.shape), (1, 16))


if __name__ == "__main__":
    unittest.main()
