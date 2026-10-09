"""检测身份映射的多数覆盖与欠分割回归。"""
import numpy as np
from celltracker.detect.labels import match_gt_labels


def test_mapping_uses_full_gt_marker_volume():
    gt=np.ones((10,),dtype=int)
    prediction=np.array([4,4,0,0,0,0,0,0,0,0])
    primary,identities=match_gt_labels(prediction,gt)
    assert primary.tolist()==[0]
    assert identities.tolist()==[[0]]


def test_undersegmentation_keeps_all_covered_markers_and_non_contiguous_ids():
    gt=np.array([2,2,2,8,8,8,0,0])
    prediction=np.array([30,30,30,30,30,30,99,99])
    primary,identities=match_gt_labels(prediction,gt,instance_ids=[99,30])
    assert primary.tolist()==[0,2]
    assert identities.tolist()==[[0,0],[2,8]]


def test_exactly_half_marker_is_unmatched():
    primary,_=match_gt_labels(np.array([3,3,0,0]),np.ones(4,dtype=int))
    assert primary.tolist()==[0]
