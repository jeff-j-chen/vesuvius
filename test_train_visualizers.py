import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from train import Trainer


class TrainerVisualizerTest(unittest.TestCase):
    @staticmethod
    def _trainer(vis_scroll_ids):
        trainer = Trainer.__new__(Trainer)
        trainer._scroll_ids = [1, 2, 3]
        trainer.c = SimpleNamespace(
            data=SimpleNamespace(vis_scroll_ids=vis_scroll_ids),
            tra=SimpleNamespace(
                n_epochs=10,
                test_int=9_999,
                test_on_final=False,
            ),
        )
        return trainer

    @patch("train.TensorboardVisualizer")
    def test_empty_visualizer_ids_disable_per_scroll_assets(self, visualizer_class):
        metrics = MagicMock()
        visualizer_class.return_value = metrics
        trainer = self._trainer([])

        trainer._init_visualizers()

        visualizer_class.assert_called_once_with(trainer.c, mode="metrics")
        self.assertEqual(trainer.scroll_vis, {})
        metrics.writer.add_scalar.assert_called_once_with("Run/Initializing", 1.0, 0)
        metrics.writer.flush.assert_called_once_with()

    @patch("train.TensorboardVisualizer")
    def test_none_visualizer_ids_preserve_all_scrolls_default(self, visualizer_class):
        metrics = MagicMock()
        visualizer_class.side_effect = [metrics, MagicMock(), MagicMock(), MagicMock()]
        trainer = self._trainer(None)

        trainer._init_visualizers()

        self.assertEqual(visualizer_class.call_count, 4)
        self.assertEqual(set(trainer.scroll_vis), {1, 2, 3})


if __name__ == "__main__":
    unittest.main()