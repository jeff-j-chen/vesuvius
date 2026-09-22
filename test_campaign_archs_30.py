import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

import campaign_archs_30 as campaign
from train import Trainer
from utils.model import ResidualConvBlock2d, create_model


class Campaign30Test(unittest.TestCase):
    def test_standard_constraints_and_single_full_resolution_exception(self):
        exceptions = []
        for test in campaign.TESTS:
            config = campaign.build_config(test)
            self.assertEqual(config.data.context_size, 192)
            self.assertEqual(config.data.depth, 8)
            self.assertTrue(config.model.surface_teacher_input)
            self.assertTrue(config.model.multitile)
            self.assertEqual(config.model.multitile_grid, 4)
            self.assertEqual(config.model.multitile_subtile, 16)
            if config.data.context_downsample == 1:
                exceptions.append(test["tid"])
            else:
                self.assertEqual(config.data.context_downsample, 2)
        self.assertEqual(exceptions, ["researcher_like"])

    def test_researcher_like_channel_ladder(self):
        test = next(test for test in campaign.TESTS if test["tid"] == "researcher_like")
        config = campaign.build_config(test)
        config.device = "cpu"
        model, _ = create_model(config)

        self.assertEqual(model.enc1.net[0].in_channels, 1)
        self.assertEqual(model.enc1.net[0].out_channels, 16)
        self.assertEqual(model.early2d_enc2.net[0].out_channels, 32)
        self.assertEqual(model.early2d_enc3.net[0].out_channels, 64)
        self.assertEqual(model.early2d_bottleneck.net[0].out_channels, 128)
        self.assertEqual(
            [block.net[0].out_channels for block in model.early2d_extra_encoders],
            [256, 320],
        )
        self.assertTrue(all(
            isinstance(block, ResidualConvBlock2d)
            for block in model.early2d_extra_encoders
        ))

    def test_pcgrad_lite_preserves_unscoped_gradient(self):
        class IdentityScaler:
            @staticmethod
            def scale(value):
                return value

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = nn.Parameter(torch.tensor([1.0, 1.0]))
                self.mid2d_head = nn.Linear(2, 1, bias=False)

        trainer = Trainer.__new__(Trainer)
        trainer.model = TinyModel()
        trainer.scaler = IdentityScaler()
        trainer.c = SimpleNamespace(tra=SimpleNamespace(
            pcgrad_lite_max_domains=4,
            pcgrad_lite_scope="head",
        ))
        head = trainer.model.mid2d_head.weight.reshape(-1)
        loss1 = trainer.model.trunk.sum() + head[0]
        loss2 = trainer.model.trunk.sum() - head[0] + head[1]
        loss = (loss1 + loss2) / 2.0

        trainer._pcgrad_lite_backward(loss, [loss1, loss2])

        torch.testing.assert_close(trainer.model.trunk.grad, torch.ones(2))
        torch.testing.assert_close(
            trainer.model.mid2d_head.weight.grad.reshape(-1),
            torch.tensor([0.25, 0.75]),
        )


if __name__ == "__main__":
    unittest.main()
