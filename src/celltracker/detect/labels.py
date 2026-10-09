"""由实例掩码和完整 GT 标记生成边监督所需的实例身份（ideas.pdf 式29）。"""

from __future__ import annotations

import numpy as np


def match_gt_labels(labels: np.ndarray, gt: np.ndarray, instance_ids=None):
    """返回主 GT 身份及全部匹配身份；零表示未匹配。

    GT 标记超过一半体素落在预测实例内时才算匹配，分母是完整标记的体积。
    欠分割实例可能覆盖多个标记，保留全部身份以生成真实前驱边；主身份取
    重叠最多者，供只接受单一身份的历史诊断使用。原文未规定像素匹配策略，
    此处采用 CTC 的多数覆盖约定作为监督数据转换规则。
    """
    labels, gt = np.asarray(labels), np.asarray(gt)
    if labels.shape != gt.shape:
        raise ValueError("预测实例和 GT 标记必须具有相同形状")
    ids = np.asarray(instance_ids if instance_ids is not None else np.unique(labels[labels > 0]), dtype=np.int64)
    primary = np.zeros(len(ids), dtype=np.int64)
    matches = [[] for _ in ids]
    row_of = {int(label): row for row, label in enumerate(ids)}
    g = gt.ravel().astype(np.int64)
    p = labels.ravel().astype(np.int64)
    sizes = np.bincount(g[g > 0])
    keep = (g > 0) & (p > 0)
    if np.any(keep):
        base = int(g.max()) + 1
        pairs, counts = np.unique(p[keep] * base + g[keep], return_counts=True)
        for ix in np.argsort(-counts, kind="stable"):
            pred_id, gt_id = divmod(int(pairs[ix]), base)
            row = row_of.get(pred_id)
            if row is not None and counts[ix] > 0.5 * sizes[gt_id]:
                matches[row].append(gt_id)
                if primary[row] == 0:
                    primary[row] = gt_id
    width = max(1, max(map(len, matches), default=0))
    all_ids = np.zeros((len(ids), width), dtype=np.int64)
    for row, values in enumerate(matches):
        all_ids[row, :len(values)] = values
    return primary, all_ids
