#!/usr/bin/env python3
"""使用冻结 nnU-Net encoder 导出每个预测实例的外观特征（ideas.pdf 式25）。

每次只读取一帧，按训练时的转轴、裁剪、归一化和重采样方式预处理，再分块前向。
在 encoder 中间层按实例质心采样，同一实例在重叠块中的向量取均值。这是将网络
特征图转换为实例向量的工程实现；网络参数和分割结果不在本步骤更新。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_centroids(centroids, transpose, bbox, cropped_shape, new_shape):
    """按体素中心坐标映射到预处理后的数组，返回 (z,y,x)。"""
    xy = np.asarray(centroids, dtype=float)[:, transpose]
    low = np.asarray(bbox, dtype=float)[:, 0]
    return (xy - low + 0.5) * (np.asarray(new_shape) / np.asarray(cropped_shape)) - 0.5


def tile_starts(shape, patch, overlap=0.5):
    starts = []
    for size, width in zip(shape, patch):
        count = max(1, math.ceil((size - width) / (width * (1 - overlap))) + 1)
        starts.append(np.unique(np.rint(np.linspace(0, size - width, count)).astype(int)))
    from itertools import product
    return product(*starts)


@torch.inference_mode()
def extract_frame(encoder, data, centroids, patch_size, stage, device):
    """逐块累计特征；GPU 中不保留完整序列或整帧高维特征图。"""
    patch = np.asarray(patch_size, dtype=int)
    shape = np.asarray(data.shape[1:])
    total_pad = np.maximum(patch - shape, 0)
    before = total_pad // 2
    data = np.pad(data, [(0, 0), *zip(before, total_pad - before)])
    xy = centroids + before
    sums, counts = None, np.zeros(len(xy), dtype=np.int32)
    for start in tile_starts(data.shape[1:], patch):
        start = np.asarray(start)
        inside = np.all((xy >= start - 0.5) & (xy < start + patch - 0.5), axis=1)
        if not np.any(inside):
            continue
        slices = tuple(slice(int(a), int(a + w)) for a, w in zip(start, patch))
        tensor = torch.from_numpy(np.ascontiguousarray(data[(slice(None), *slices)])).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            maps = encoder(tensor)
        fmap = maps[stage] if isinstance(maps, (list, tuple)) else maps
        # grid_sample 接受 (x,y,z)，归一化坐标 −1/1 表示块边界。
        normalized = 2 * (xy[inside] - start + 0.5) / patch - 1
        grid = torch.as_tensor(normalized[:, ::-1].copy(), dtype=torch.float32, device=device).view(1, -1, 1, 1, 3)
        values = F.grid_sample(fmap.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
        values = values[0, :, :, 0, 0].T.cpu().numpy()
        if sums is None:
            sums = np.zeros((len(xy), values.shape[1]), dtype=np.float64)
        sums[inside] += values
        counts[inside] += 1
        del tensor, maps, fmap, values
    if len(xy) and (sums is None or np.any(counts == 0)):
        raise RuntimeError("存在未被任何推理块覆盖的实例")
    return (sums / counts[:, None]).astype(np.float32) if sums is not None else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--images", required=True, help="nnU-Net imagesTs_eval 目录")
    ap.add_argument("--model", required=True, help="包含 plans.json 和 fold_0 的模型目录")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()
    os.environ["nnUNet_compile"] = "false"
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("要求 CUDA，但当前环境不可用")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    parts = out.with_suffix(".frames")
    parts.mkdir(parents=True, exist_ok=True)
    meta = {"schema": "encoder_centroid_v1", "model_sha256": file_hash(Path(args.model)/"fold_0"/args.checkpoint),
            "detection_sha256": file_hash(args.h5), "stage": args.stage, "seq": args.seq,
            "aggregation": "mean_of_overlapping_tile_centroid_samples"}
    manifest = parts / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != meta:
        raise ValueError("恢复目录与本次模型/检测输入不一致")
    manifest.write_text(json.dumps(meta, indent=2))
    predictor = nnUNetPredictor(device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(args.model, (0,), args.checkpoint)
    network = predictor.network.to(device).eval().requires_grad_(False)
    encoder = network.encoder
    cm, pm = predictor.configuration_manager, predictor.plans_manager
    preprocessor = cm.preprocessor_class(verbose=False)
    outputs, feature_dim = {}, None
    started = time.time()
    with h5py.File(args.h5, "r") as hf:
        keys = sorted(hf["frames"])
        if args.max_frames:
            keys = keys[:args.max_frames]
        for index, key in enumerate(keys):
            t = int(key)
            cached = parts / f"frame_{t:04d}.npz"
            group = hf["frames"][key]
            if cached.exists():
                with np.load(cached) as saved:
                    item = {name: saved[name] for name in saved.files}
            else:
                image = Path(args.images) / f"CE{args.seq}_f{t:03d}_0000.nii.gz"
                data, _, prop = preprocessor.run_case([str(image)], None, pm, cm, predictor.dataset_json)
                cents = group["centroid"][:]
                transformed = transform_centroids(cents, pm.transpose_forward, prop["bbox_used_for_cropping"],
                    prop["shape_after_cropping_and_before_resampling"], data.shape[1:])
                feat = extract_frame(encoder, data, transformed, cm.patch_size, args.stage, device)
                if feat is None:
                    if feature_dim is None:
                        raise ValueError("首帧无实例，无法推断特征维度")
                    feat = np.empty((0, feature_dim), dtype=np.float32)
                item = {f"frame_{t}": feat, f"centroid_{t}": cents, f"label_{t}": group["label"][:]}
                np.savez_compressed(cached, **item)
                del data
            feature_dim = item[f"frame_{t}"].shape[1]
            outputs.update(item)
            elapsed = time.time() - started
            print(json.dumps({"frame": t, "done": index+1, "total": len(keys), "feature_dim": feature_dim,
                              "elapsed_s": round(elapsed, 1), "gpu_allocated_mb": round(torch.cuda.memory_allocated()/2**20, 1) if device.type=="cuda" else 0}), flush=True)
    meta["feature_dim"] = feature_dim
    outputs["metadata_json"] = np.array(json.dumps(meta))
    np.savez_compressed(out, **outputs)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print("complete", str(out), out.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
