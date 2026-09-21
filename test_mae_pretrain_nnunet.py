import unittest

import numpy as np
import torch

from mae_pretrain_nnunet import CropSampler, make_physical_sampler
from utils.config import DEFAULT_SCROLLS, DEFAULT_TEST_SCROLL_IDS


class CropSamplerTest(unittest.TestCase):
    def test_single_occupied_block_has_train_and_monitor_anchors(self):
        sampler = CropSampler.__new__(CropSampler)
        sampler.scroll_id = 20231205222200
        sampler.ctx = 64
        sampler.block_px = 512
        sampler.y0 = 0
        sampler.y1 = 640
        sampler.x0 = 0
        sampler.x1 = 640
        mask = np.zeros((640, 640), dtype=np.uint8)
        mask[:400, :400] = 1
        sampler.mask_integral = np.pad(
            mask.cumsum(axis=0).cumsum(axis=1),
            ((1, 0), (1, 0)),
        )

        train, monitor = sampler._build_candidate_coords(0.1)

        self.assertGreater(len(train), 0)
        self.assertGreater(len(monitor), 0)
        self.assertFalse(set(map(tuple, train)) & set(map(tuple, monitor)))


class PhysicalSamplerTest(unittest.TestCase):
    def test_round_robin_balances_physical_scrolls_then_segments(self):
        class StubSampler:
            def __init__(self, scroll_id):
                self.scroll_id = scroll_id
                self.samples = 0

            def sample(self, count, rng):
                self.samples += count
                return torch.zeros((count, 1))

        scroll_ids = [int(scroll.scroll_id) for scroll in DEFAULT_SCROLLS]
        scroll_ids.extend(DEFAULT_TEST_SCROLL_IDS)
        samplers = [StubSampler(scroll_id) for scroll_id in scroll_ids]

        sampler, physical_count = make_physical_sampler(samplers)
        batch = sampler.sample(physical_count * 3, np.random.default_rng(0))

        self.assertEqual(physical_count, 17)
        self.assertEqual(batch.shape, (physical_count * 3, 1))
        by_id = {item.scroll_id: item.samples for item in samplers}
        self.assertEqual(
            [by_id[scroll_id] for scroll_id in (20260115000000, 20260317000000, 20250223000000)],
            [1, 1, 1],
        )
        self.assertTrue(all(by_id[scroll_id] == 3 for scroll_id in DEFAULT_TEST_SCROLL_IDS))


if __name__ == "__main__":
    unittest.main()