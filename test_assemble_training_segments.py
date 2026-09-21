import inspect
import unittest

import assemble_training_segments as assembly


class PooledSurfaceConfigTest(unittest.TestCase):
    def test_dispatch_arguments_match_helper_signature(self):
        inspect.signature(assembly._assemble_pooled_surface).bind(
            "segment",
            "zarr-id",
            {},
            8,
            64,
            64,
            expected_shape=(109, 1, 1),
            force=False,
        )

    def test_pooled_source_shapes_are_pinned(self):
        expected_shapes = {
            "w018": (109, 10595, 24525),
            "paris4": (109, 12750, 9995),
        }
        for name, expected_shape in expected_shapes.items():
            with self.subTest(name=name):
                options = assembly.FRAG_OPTS[name]
                self.assertTrue(options["pooled_special"])
                self.assertEqual(options["surface_expected_shape"], expected_shape)


if __name__ == "__main__":
    unittest.main()