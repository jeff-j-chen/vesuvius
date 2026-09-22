import unittest

import campaign_archs_30 as campaign30
import campaign_archs_31 as campaign
from utils.model import ConvBlock2d, ResidualConvBlock2d, create_model
from utils.training_utils import calibrate_character_threshold


class Campaign31Test(unittest.TestCase):
    def test_requested_arm_matrix(self):
        self.assertEqual(
            {test["tid"] for test in campaign.TESTS},
            {
                "mid_control_seed42",
                "mid_groupdro_anchor",
                "mid_groupdro_anchor_ema",
                "mid_domain_mean",
                "mid_pcgrad_gram_exact",
                "mid_pcgrad_gram_sparse4",
                "mid_depth12",
                "mid_overlap8",
                "mid_overlap12",
                "early_overlap12",
                "early_deep_residual_groupdro_anchor",
                "early_raw_instance_groupdro",
                "early_raw_instance_groupdro_anchor",
                "mid_wide15_groupdro",
                "early_gated_patch_groupdro",
                "early_residual_depth3",
                "early_residual_extra320",
                "early_deep_nonresidual",
                "early_deep_residual_replica",
                "early_gated_instance",
                "early_raw_ibn",
                "researcher_ds2_full",
                "researcher_full",
            },
        )

    def test_mid_combination_factorial(self):
        expected = {
            "mid_control_seed42": (False, 0.0, False, 42),
            "mid_groupdro_anchor": (True, 0.001, False, 41),
            "mid_groupdro_anchor_ema": (True, 0.001, True, 41),
        }
        for tid, values in expected.items():
            test = next(test for test in campaign.TESTS if test["tid"] == tid)
            config = campaign.build_config(test)
            self.assertEqual(
                (
                    config.tra.physical_domain_groupdro,
                    config.tra.mae_anchor_lambda,
                    config.tra.model_ema,
                    config.tra.seed,
                ),
                values,
            )
            self.assertFalse(config.tra.pcgrad_gram)
            self.assertFalse(config.tra.physical_patch_groupdro)
            self.assertTrue(config.model.mid_2d_unet)
            self.assertTrue(config.model.gated_stems)

    def test_all_arms_retain_campaign30_protocol(self):
        for test in campaign.TESTS:
            config = campaign.build_config(test)
            self.assertEqual(config.data.context_size, 192)
            self.assertEqual(config.data.depth, int(test["depth"]))
            self.assertEqual(config.data.depth, 12 if "12" in test["tid"] else 8)
            self.assertEqual(config.dl.batch_size, campaign.FINETUNE_BATCH_SIZE)
            self.assertEqual(config.tra.lr, campaign.FINETUNE_LR)
            if config.data.context_downsample == 1:
                self.assertTrue(config.model.early_2d_unet)
                self.assertFalse(config.model.mid_2d_unet)
            else:
                self.assertEqual(config.data.context_downsample, 2)
            self.assertTrue(config.model.surface_teacher_input)
            self.assertTrue(config.model.multitile)
            self.assertEqual(config.model.multitile_grid, 4)
            self.assertEqual(config.model.multitile_subtile, 16)
            self.assertTrue(config.model.require_architecture_init)
            self.assertTrue(config.tra.character_calibrate_threshold)
            self.assertLessEqual(
                int(config.tra.physical_domain_groupdro)
                + int(config.tra.physical_patch_groupdro),
                1,
            )

    def test_exact_architectures_reuse_campaign30_pretraining(self):
        expected = {
            "mid_gated_c30": "mid_gated_c29",
            "early_deep_residual_c30": "early_deep_residual2d",
            "early_raw_instance_c30": "early_raw_instance",
            "mid_wide15_c30": "mid_wide15_gated",
            "early_gated_c30": "early_gated",
        }
        for key, source in expected.items():
            self.assertEqual(
                campaign._pretrain_path(key),
                campaign30._pretrain_path(source),
            )
        self.assertEqual(
            {
                key for key, spec in campaign.PRETRAIN_SPECS.items()
                if not spec.get("reuse_campaign30")
            },
            {
                "early_residual_depth3",
                "early_residual_extra320",
                "early_deep_nonresidual",
                "early_gated_instance",
                "early_raw_ibn",

                "researcher_ds2_full",
                "researcher_full",
                "mid_overlap12_c29",
                "mid_depth12",
                "mid_overlap8",
                "early_overlap12",
            },
        )

    def test_overlap_depth12_decomposition(self):
        import campaign_archs_29 as campaign29

        expected = {
            "mid_depth12": (12, False, 0),
            "mid_overlap8": (8, True, 3),
            "mid_overlap12": (12, True, 5),
            "early_overlap12": (12, True, 5),
        }
        for tid, (depth, overlap, windows) in expected.items():
            test = next(t for t in campaign.TESTS if t["tid"] == tid)
            config = campaign.build_config(test)
            config.device = "cpu"
            model, _ = create_model(config)
            self.assertEqual(config.data.depth, depth)
            self.assertEqual(config.model.overlapping_depth_windows, overlap)
            self.assertEqual(model._overlap_window_count, windows)
            self.assertEqual(config.model.mid_2d_unet, not tid.startswith("early_"))
            self.assertEqual(config.model.early_2d_unet, tid.startswith("early_"))
            self.assertTrue(config.model.gated_stems)
            self.assertEqual(config.model.norm_mode, "ibn_full")
            self.assertFalse(config.tra.physical_domain_groupdro or config.tra.pcgrad_gram)
            spec = campaign.PRETRAIN_SPECS[test["pretrain_key"]]
            if "reuse_campaign29" not in spec:
                self.assertEqual(spec["depth"], depth)
        self.assertEqual(
            campaign._pretrain_path("mid_overlap12_c29"),
            campaign29._pretrain_path("mid_gated_overlap12_all"),
        )
        spec = campaign.PRETRAIN_SPECS["mid_depth12"]
        self.assertEqual((spec["d_start"], spec["d_end"]), (8, 20))

    def test_pcgrad_gram_arms(self):
        expected = {
            "mid_domain_mean": (0, 0.0),
            "mid_pcgrad_gram_exact": (1, 0.0),
            "mid_pcgrad_gram_sparse4": (4, 0.8),
        }
        for tid, (interval, ema) in expected.items():
            config = campaign.build_config(next(t for t in campaign.TESTS if t["tid"] == tid))
            self.assertTrue(config.tra.pcgrad_gram)
            self.assertEqual(config.tra.pcgrad_gram_interval, interval)
            self.assertEqual(config.tra.pcgrad_gram_ema, ema)
            self.assertFalse(config.tra.pcgrad or config.tra.pcgrad_lite)
            self.assertFalse(config.tra.physical_domain_groupdro)
            self.assertEqual(config.tra.mae_anchor_lambda, 0.0)
            self.assertFalse(config.tra.model_ema)
            self.assertTrue(config.model.mid_2d_unet)

    def test_pcgrad_coefficients_match_explicit_projection(self):
        import torch
        from train import pcgrad_coefficients

        for seed in range(20):
            torch.manual_seed(seed)
            gradients = torch.randn(6, 50, dtype=torch.float64)
            gradients[1] = -0.8 * gradients[0] + 0.2 * gradients[1]
            gram = (gradients @ gradients.T).tolist()
            torch.manual_seed(1000 + seed)
            weights = torch.tensor(pcgrad_coefficients(gram), dtype=torch.float64)
            torch.manual_seed(1000 + seed)
            projected = []
            for task in range(6):
                adjusted = gradients[task].clone()
                for other in torch.randperm(6).tolist():
                    if other == task:
                        continue
                    dot = adjusted @ gradients[other]
                    if dot < 0:
                        adjusted -= dot / gradients[other].square().sum() * gradients[other]
                projected.append(adjusted)
            torch.testing.assert_close(weights @ gradients, torch.stack(projected).mean(0))
            self.assertTrue(bool((weights >= 1.0 / 6 - 1e-12).all()))

    def test_pcgrad_coefficients_without_conflict_are_equal(self):
        from train import pcgrad_coefficients

        gram = [[1.0, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 0.5]]
        self.assertEqual(pcgrad_coefficients(gram), [1.0 / 3] * 3)

    def test_early_depth_decomposition(self):
        cases = {
            "early_residual_depth3": (True, 3, (), ResidualConvBlock2d),
            "early_residual_extra320": (True, 2, (320,), ResidualConvBlock2d),
            "early_deep_nonresidual": (False, 3, (320,), ConvBlock2d),
            "early_deep_residual_replica": (True, 3, (320,), ResidualConvBlock2d),
        }
        for tid, expected in cases.items():
            test = next(test for test in campaign.TESTS if test["tid"] == tid)
            config = campaign.build_config(test)
            config.device = "cpu"
            model, _ = create_model(config)
            residual, depth, extra_channels, block_type = expected
            self.assertTrue(config.model.early_2d_unet)
            self.assertFalse(config.model.mid_2d_unet)
            self.assertEqual(config.model.residual_2d_unet, residual)
            self.assertEqual(config.model.two_d_block_depth, depth)
            self.assertEqual(config.model.two_d_extra_channels, extra_channels)
            self.assertIsInstance(model.early2d_enc2, block_type)
            self.assertEqual(
                len(model.early2d_extra_encoders),
                len(extra_channels),
            )

    def test_early_raw_instance_controls(self):
        gated_instance = campaign.build_config(next(
            test for test in campaign.TESTS if test["tid"] == "early_gated_instance"
        ))
        raw_ibn = campaign.build_config(next(
            test for test in campaign.TESTS if test["tid"] == "early_raw_ibn"
        ))
        self.assertTrue(gated_instance.model.gated_stems)
        self.assertFalse(gated_instance.model.raw_only_stem)
        self.assertEqual(gated_instance.model.norm_mode, "instance")
        self.assertFalse(raw_ibn.model.gated_stems)
        self.assertTrue(raw_ibn.model.raw_only_stem)
        self.assertEqual(raw_ibn.model.norm_mode, "ibn_full")

    def test_researcher_like_bundles_multiple_changes(self):
        researcher_test = next(
            test for test in campaign30.TESTS if test["tid"] == "researcher_like"
        )
        early_raw_test = next(
            test for test in campaign30.TESTS if test["tid"] == "early_raw_instance"
        )
        researcher = campaign30.build_config(researcher_test)
        early_raw = campaign30.build_config(early_raw_test)
        self.assertEqual(researcher.data.context_size, early_raw.data.context_size)
        self.assertEqual(researcher.data.context_downsample, 1)
        self.assertEqual(early_raw.data.context_downsample, 2)
        self.assertEqual(researcher.dl.batch_size, 4)
        self.assertEqual(early_raw.dl.batch_size, 96)
        self.assertEqual(researcher.model.channels_mult, 0.5)
        self.assertEqual(early_raw.model.channels_mult, 1.0)
        self.assertTrue(researcher.model.residual_2d_unet)
        self.assertFalse(early_raw.model.residual_2d_unet)
        self.assertEqual(researcher.model.two_d_extra_channels, (256, 320))
        self.assertEqual(early_raw.model.two_d_extra_channels, ())
        self.assertNotEqual(researcher.init_weights, early_raw.init_weights)

    def test_researcher_isolation_matrix(self):
        expected = {
            "researcher_ds2_full": (2, 0.5, True, (256, 320)),
            "researcher_full": (1, 0.5, True, (256, 320)),
        }
        for tid, values in expected.items():
            test = next(test for test in campaign.TESTS if test["tid"] == tid)
            config = campaign.build_config(test)
            self.assertEqual(
                (
                    config.data.context_downsample,
                    config.model.channels_mult,
                    config.model.residual_2d_unet,
                    config.model.two_d_extra_channels,
                ),
                values,
            )
            self.assertEqual(config.dl.batch_size, campaign.FINETUNE_BATCH_SIZE)
            self.assertEqual(config.tra.lr, campaign.FINETUNE_LR)
            self.assertTrue(config.model.early_2d_unet)
            self.assertFalse(config.model.mid_2d_unet)
            self.assertTrue(config.model.raw_only_stem)
            self.assertFalse(config.model.gated_stems)
            self.assertEqual(config.model.norm_mode, "instance")

    def test_researcher_isolation_does_not_duplicate_campaign30(self):
        fields = (
            "context_downsample",
            "channels_mult",
            "residual_2d_unet",
            "two_d_extra_channels",
        )

        def signature(test):
            return tuple(test[field] for field in fields)

        campaign30_signatures = {
            signature(test)
            for test in campaign30.TESTS
            if test["early_2d_unet"]
            and test["raw_only_stem"]
            and test["norm_mode"] == "instance"
        }
        for test in campaign.TESTS:
            if not test["tid"].startswith("researcher_"):
                continue
            if test["tid"] == "researcher_full":
                self.assertIn(signature(test), campaign30_signatures)
            else:
                self.assertNotIn(signature(test), campaign30_signatures)

    def test_batch_and_learning_rate_are_never_decoupled(self):
        for test in campaign.TESTS:
            self.assertNotIn("batch_size", test)
            config = campaign.build_config(test)
            self.assertEqual(
                (config.dl.batch_size, config.tra.lr),
                (campaign.FINETUNE_BATCH_SIZE, campaign.FINETUNE_LR),
            )
        for spec in campaign.PRETRAIN_SPECS.values():
            self.assertNotIn("batch_size", spec)
            self.assertNotIn("accum_steps", spec)

    def test_character_threshold_calibration(self):
        metrics = calibrate_character_threshold(
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0.1, 0.6, 0.2, 0.7, 0.15, 0.65, 0.25, 0.75],
            [1, 1, 1, 1, 2, 2, 2, 2],
        )
        self.assertEqual(metrics["character_calibrated_f1_macro"], 1.0)
        self.assertEqual(metrics["character_calibrated_recall_macro"], 1.0)
        self.assertEqual(metrics["character_calibrated_ring_fpr_macro"], 0.0)
        self.assertEqual(metrics["character_calibrated_success_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
