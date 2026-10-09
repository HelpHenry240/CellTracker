"""训练图与推理输入的可复现契约，避免跨检测来源或特征配置误用权重。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pipeline_contract(cfg, detection_meta=None, encoder_meta=None):
    data = asdict(cfg)
    selected = {key:data[key] for key in ("measure","coupling","motion","multiscale","graph","node")}
    selected["node"].pop("encoder_feat_path",None)
    if cfg.motion.enabled:
        selected["motion_initialization"] = data["reconstruct"]
    # 实际物理间距写入契约；侧车路径及序列名可以不同，网络和特征含义必须一致。
    meta = detection_meta or {}
    if "spacing_zyx" in meta:
        selected["measure"]["spacing_zyx"] = [round(float(v),7) for v in meta["spacing_zyx"]]
    selected.update(schema=cfg.schema_version,detection_source=cfg.detection_source,
                    label_mapping=str(meta.get("label_mapping","unspecified")))
    if encoder_meta:
        selected["encoder"] = {k:encoder_meta.get(k) for k in ("schema","model_sha256","stage","feature_dim","aggregation")}
    encoded = json.dumps(selected,sort_keys=True,separators=(",",":"))
    return {"signature":hashlib.sha256(encoded.encode()).hexdigest(),"definition":selected}


def check_contract(expected, actual):
    if not expected or not actual or expected.get("signature") != actual.get("signature"):
        raise ValueError("GNN 权重与当前检测/图配置不一致；请为该配置独立建图和训练")
