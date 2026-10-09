"""多对一欠分割匹配不应把精确率抬到 1 以上。"""
import numpy as np
import pytest
from celltracker.eval.local_metrics import StreamingDiagnostics,tracking_diagnostics

@pytest.mark.parametrize('streaming',[False,True])
def test_undersegmented_precision_counts_result_identities(streaming):
    gt=np.array([1,1,2,2,0,0])
    res=np.array([8,8,8,8,9,9])
    if streaming:
        diag=StreamingDiagnostics(); diag.add_frame(0,gt,res); result=diag.result()
    else:
        result=tracking_diagnostics({0:gt},{0:res})
    assert result['detection_precision']==.5
    assert result['detection_recall']==1
    assert result['merged_gt_excess']==1
    assert result['matched_results']==1
