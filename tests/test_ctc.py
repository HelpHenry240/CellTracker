"""CTC 读取模块的单元测试（用合成数据，不依赖真实数据集）。"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from celltracker.data.ctc import (
    CTCSequence,
    list_sequences,
    object_table_from_labels,
    read_man_track,
)


def _make_dataset(root, seq="01", ndim=3):
    """构造一个最小可用的合成 CTC 数据集目录。"""
    img_dir = root / seq
    tra_dir = root / f"{seq}_GT" / "TRA"
    seg_dir = root / f"{seq}_GT" / "SEG"
    for d in (img_dir, tra_dir, seg_dir):
        d.mkdir(parents=True, exist_ok=True)

    shape = (8, 16, 16) if ndim == 3 else (16, 16)
    rng = np.random.default_rng(0)

    for t in range(2):
        img = (rng.random(shape) * 100).astype(np.uint16)
        labels = np.zeros(shape, dtype=np.uint16)
        # track 1: 2x2x2 方块，向下漂移；track 2: 3x3x3 方块
        sl = (slice(1, 3), slice(2, 4), slice(2, 4)) if ndim == 3 else (slice(1, 3), slice(2, 4))
        labels[sl] = 1 + t
        sl2 = (slice(5, 8), slice(9, 12), slice(9, 12)) if ndim == 3 else (slice(5, 8), slice(9, 12))
        labels[sl2] = 10
        tifffile.imwrite(img_dir / f"t{t:03d}.tif", img)
        tifffile.imwrite(tra_dir / f"man_track{t:03d}.tif", labels)
        tifffile.imwrite(seg_dir / f"man_seg{t:03d}.tif", (labels > 0).astype(np.uint16))

    (tra_dir / "man_track.txt").write_text("1 0 0 0\n2 1 1 0\n10 0 1 0\n")
    return root


@pytest.fixture()
def ds(tmp_path):
    root = tmp_path / "Fluo-N3DH-FAKE"
    _make_dataset(root, "01", ndim=3)
    return root


def test_read_man_track(ds):
    tracks = read_man_track(ds / "01_GT" / "TRA" / "man_track.txt")
    assert set(tracks) == {1, 2, 10}
    assert tracks[2].parent == 0 and tracks[2].begin == 1
    assert tracks[1].length == 1


def test_list_sequences(ds):
    assert list_sequences(ds) == ["01"]


def test_sequence_discovery_and_shape(ds):
    s = CTCSequence(ds, "01")
    assert s.n_frames == 2
    assert s.frames == [0, 1]
    assert s.shape == (8, 16, 16)
    assert s.ndim == 3
    assert s.has_tra and s.has_seg
    assert s.seg_frames == [0, 1]


def test_object_table_centroid_volume(ds):
    s = CTCSequence(ds, "01")
    tab = s.object_table(0)
    labels = tab["label"]
    assert set(labels.tolist()) == {1, 10}
    i1 = int(np.where(labels == 1)[0][0])
    i10 = int(np.where(labels == 10)[0][0])
    # track 1 = 2x2x2 = 8 体素，质心 (1.5, 2.5, 2.5)
    assert tab["volume"][i1] == 8
    np.testing.assert_allclose(tab["centroid"][i1], [1.5, 2.5, 2.5])
    # track 10 = 3x3x3 = 27 体素，质心 (6, 10, 10)
    assert tab["volume"][i10] == 27
    np.testing.assert_allclose(tab["centroid"][i10], [6.0, 10.0, 10.0])
    # 包围盒
    np.testing.assert_array_equal(tab["bbox_min"][i1], [1, 2, 2])
    np.testing.assert_array_equal(tab["bbox_max"][i1], [2, 3, 3])


def test_object_table_empty_and_2d(tmp_path):
    # 空帧
    empty = np.zeros((4, 4), dtype=np.uint16)
    tab = object_table_from_labels(empty)
    assert tab["label"].size == 0 and tab["centroid"].shape == (0, 2)

    # 2D 数据集
    root = tmp_path / "Fluo-N2DL-FAKE"
    _make_dataset(root, "01", ndim=2)
    s = CTCSequence(root, "01")
    assert s.ndim == 2 and s.shape == (16, 16)
    tab2 = s.object_table(0)
    assert tab2["centroid"].shape[1] == 2


def test_object_table_intensity_stats(ds):
    s = CTCSequence(ds, "01")
    img = s.image(0)
    tab = s.object_table(0)
    i1 = int(np.where(tab["label"] == 1)[0][0])
    expected = img[1:3, 2:4, 2:4].astype(float)
    assert tab["intensity_mean"][i1] == pytest.approx(expected.mean(), rel=1e-5)
    assert tab["intensity_std"][i1] == pytest.approx(expected.std(), rel=1e-5)
