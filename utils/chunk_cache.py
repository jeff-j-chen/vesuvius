"""Campaign-lifetime selective cache for local zarr volume chunks."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np


class SelectiveChunkVolume:
    """Read-only zarr-like volume backed by selected full-depth spatial chunks."""

    def __init__(self, path: str, shape, dtype, chunks, norm_stats=None):
        self.path = os.path.abspath(path)
        self.shape = tuple(int(value) for value in shape)
        self.source_dtype = np.dtype(dtype)
        self.norm_stats = None if norm_stats is None else tuple(float(value) for value in norm_stats)
        self.normalized = self.norm_stats is not None
        self.dtype = np.dtype(np.float32) if self.normalized else self.source_dtype
        self.chunks = tuple(int(value) for value in chunks)
        self.ndim = 3
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self._source = None
        self.cache_misses = 0

    @property
    def cached_nbytes(self) -> int:
        return sum(array.nbytes for array in self._cache.values())

    def _open_source(self):
        if self._source is None:
            import zarr

            self._source = zarr.open(self.path, mode="r")
        return self._source

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_source"] = None
        return state

    def _prepare_array(self, array) -> np.ndarray:
        array = np.asarray(array)
        if self.normalized:
            mean, std, lower, upper = self.norm_stats
            array = array.astype(np.float32)
            if std != 0:
                array = (array - mean) / std
                array = (array - lower) / (upper - lower)
                array = np.clip(array, 0.0, 1.0)
        return np.ascontiguousarray(array, dtype=self.dtype)

    def preload(self, spatial_chunks: Iterable[tuple[int, int]], source, workers: int = 8) -> None:
        requested = sorted(set(spatial_chunks))
        missing = [key for key in requested if key not in self._cache]
        if not missing:
            print(
                f"[selective-cache] reuse {os.path.basename(self.path)}: "
                f"{len(requested):,} chunks {self.cached_nbytes / 1024**3:.2f} GiB",
                flush=True,
            )
            return

        chunk_y, chunk_x = self.chunks[1], self.chunks[2]
        height, width = self.shape[1], self.shape[2]

        def load(key: tuple[int, int]):
            cy, cx = key
            y0, x0 = cy * chunk_y, cx * chunk_x
            y1, x1 = min(y0 + chunk_y, height), min(x0 + chunk_x, width)
            array = self._prepare_array(source[:, y0:y1, x0:x1])
            array.setflags(write=False)
            return key, array

        worker_count = max(1, min(int(workers), len(missing)))
        print(
            f"[selective-cache] preload {os.path.basename(self.path)}: "
            f"new={len(missing):,} requested={len(requested):,} workers={worker_count}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for index, (key, array) in enumerate(executor.map(load, missing), start=1):
                self._cache[key] = array
                if index % 1000 == 0 or index == len(missing):
                    print(
                        f"[selective-cache] {os.path.basename(self.path)} "
                        f"{index:,}/{len(missing):,}",
                        flush=True,
                    )
        print(
            f"[selective-cache] ready {os.path.basename(self.path)}: "
            f"chunks={len(self._cache):,} ram={self.cached_nbytes / 1024**3:.2f} GiB",
            flush=True,
        )

    @staticmethod
    def _unit_slice(value, size: int) -> slice:
        if not isinstance(value, slice):
            raise TypeError("selective chunk volumes require slice indexing")
        start, stop, step = value.indices(size)
        if step != 1:
            raise ValueError("selective chunk volumes require unit-stride slices")
        return slice(start, stop)

    def __getitem__(self, item):
        if not isinstance(item, tuple) or len(item) != 3:
            raise TypeError("selective chunk volumes require three-dimensional slicing")
        z_slice = self._unit_slice(item[0], self.shape[0])
        y_slice = self._unit_slice(item[1], self.shape[1])
        x_slice = self._unit_slice(item[2], self.shape[2])
        out_shape = (
            max(0, z_slice.stop - z_slice.start),
            max(0, y_slice.stop - y_slice.start),
            max(0, x_slice.stop - x_slice.start),
        )
        if any(size == 0 for size in out_shape):
            return np.empty(out_shape, dtype=self.dtype)

        chunk_y, chunk_x = self.chunks[1], self.chunks[2]
        cy0, cy1 = y_slice.start // chunk_y, (y_slice.stop - 1) // chunk_y
        cx0, cx1 = x_slice.start // chunk_x, (x_slice.stop - 1) // chunk_x
        required = [
            (cy, cx)
            for cy in range(cy0, cy1 + 1)
            for cx in range(cx0, cx1 + 1)
        ]
        missing = [key for key in required if key not in self._cache]
        if missing:
            self.cache_misses += 1
            if self.cache_misses <= 3:
                print(
                    f"[selective-cache] WARNING {os.path.basename(self.path)} "
                    f"miss={missing[:4]} slice={(z_slice, y_slice, x_slice)}; using zarr fallback",
                    flush=True,
                )
            source = self._open_source()
            return self._prepare_array(source[z_slice, y_slice, x_slice])

        output = np.empty(out_shape, dtype=self.dtype)
        for cy, cx in required:
            cached = self._cache[(cy, cx)]
            global_y0, global_x0 = cy * chunk_y, cx * chunk_x
            ys, ye = max(y_slice.start, global_y0), min(y_slice.stop, global_y0 + cached.shape[1])
            xs, xe = max(x_slice.start, global_x0), min(x_slice.stop, global_x0 + cached.shape[2])
            output[
                :,
                ys - y_slice.start:ye - y_slice.start,
                xs - x_slice.start:xe - x_slice.start,
            ] = cached[
                z_slice,
                ys - global_y0:ye - global_y0,
                xs - global_x0:xe - global_x0,
            ]
        return output


_CACHE_REGISTRY: dict[str, SelectiveChunkVolume] = {}


def get_selective_chunk_volume(source, norm_stats=None) -> SelectiveChunkVolume:
    path = os.path.abspath(str(source.store.path))
    cached = _CACHE_REGISTRY.get(path)
    if cached is None:
        cached = SelectiveChunkVolume(
            path,
            source.shape,
            source.dtype,
            source.chunks,
            norm_stats=norm_stats,
        )
        _CACHE_REGISTRY[path] = cached
    elif (
        cached.shape != tuple(source.shape)
        or cached.source_dtype != np.dtype(source.dtype)
        or cached.chunks != tuple(source.chunks)
        or cached.norm_stats != (
            None if norm_stats is None else tuple(float(value) for value in norm_stats)
        )
    ):
        raise RuntimeError(f"selective cache metadata changed for {path}")
    return cached
