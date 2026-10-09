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
    if cfg.schema_version == 'ideas-v3' and cfg.detection_source == 'nnunet_pred':
        # 同为预测实例也可能来自不同拆分流程。前端参数改变后，GNN 必须重训。
        required=('h_frac','min_volume','gaussian_sigma','watershed','oracle_markers','resplit_k')
        missing=[key for key in required if key not in meta]
        if missing:
            raise ValueError(f'预测检测缺少前端来源参数：{missing}')
        names=(*required,'unseeded_policy','foreground_source')
        frontend={key:meta.get(key) for key in names}
        frontend={key:value.item() if hasattr(value,'item') else value for key,value in frontend.items()}
        if float(frontend['h_frac'])<=0:
            if 'min_distance' not in meta:
                raise ValueError('固定距离种子缺少 min_distance')
            frontend['min_distance']=int(meta['min_distance'])
        frontend['resplit_version']=str(meta.get('resplit_version','legacy')) if float(frontend['resplit_k'])>0 else 'disabled'
        if bool(meta['oracle_markers']):
            # Oracle种子的身份转换改变了检测实例；旧连通域版本的GNN不能直接复用。
            frontend['oracle_seed_version']=str(meta.get('oracle_seed_version','legacy_connected_components'))
        selected['frontend']=frontend
    if encoder_meta:
        selected["encoder"] = {k:encoder_meta.get(k) for k in ("schema","model_sha256","stage","feature_dim","aggregation")}
    encoded = json.dumps(selected,sort_keys=True,separators=(",",":"))
    return {"signature":hashlib.sha256(encoded.encode()).hexdigest(),"definition":selected}


def check_contract(expected, actual):
    if not expected or not actual or expected.get("signature") != actual.get("signature"):
        raise ValueError("GNN 权重与当前检测/图配置不一致；请为该配置独立建图和训练")
