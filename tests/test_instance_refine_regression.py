"""再切分中的背景编号必须保持为零，不能并到既有实例中。"""
import numpy as np
import importlib.util
from pathlib import Path
import h5py
import tifffile
import pytest
from celltracker.detect import instances


def test_resplit_does_not_offset_background_into_existing_instance(monkeypatch):
    labels=np.zeros((5,12,12),dtype=np.int32)
    labels[1:4,1:8,1:8]=1
    labels[1:3,9:11,1:3]=2
    labels[1:3,9:11,8:10]=3
    def fake_split(region,cfg):
        result=np.zeros(region.shape,dtype=np.int32)
        z,y,x=np.where(region)
        result[z,y,x]=np.where(x<x.mean(),1,2)
        result[z[0],y[0],x[0]]=0
        return result
    monkeypatch.setattr(instances,'split_instances',fake_split)
    refined=instances.refine_oversized_instances(labels>0,labels,instances.InstanceSplitConfig(),k=1.6)
    assert refined[1,1,1]==0
    # 两个未再切的小实例各自保持原体积；编号可因压缩而改变。
    for original_id in [2,3]:
        values=np.unique(refined[labels==original_id])
        assert len(values)==1 and values[0]>0
        assert np.sum(refined==values[0])==np.sum(labels==original_id)


def test_resplit_off_keeps_labels_and_recomputes_raw_image_features(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/resplit_detections.py'
    spec=importlib.util.spec_from_file_location('resplit_script',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    image_dir=tmp_path/'images'; image_dir.mkdir()
    labels=np.zeros((4,8,8),dtype=np.int32)
    labels[1:3,1:3,1:3]=1; labels[1:3,5:7,5:7]=2
    original=tmp_path/'input.h5'
    with h5py.File(original,'w') as f:
        f.attrs.update(spacing_zyx=[1.,.2,.2],h_frac=.1,min_volume=1,gaussian_sigma=0.,resplit_k=0.)
        frames=f.create_group('frames')
        for t in range(2):
            g=frames.create_group(f'{t:04d}')
            g.create_dataset('labels',data=labels);g.create_dataset('label',data=[1,2])
            tifffile.imwrite(image_dir/f't{t:03d}.tif',np.where(labels==1,10,np.where(labels==2,100,0)).astype('uint16'),photometric='minisblack')
    before=module.sha256(original)
    output=tmp_path/'output.h5'
    report=module.resplit(original,original,image_dir,output,k=0)
    assert module.sha256(original)==before and len(report['frames'])==2
    with h5py.File(output) as f:
        np.testing.assert_array_equal(f['frames/0000/labels'],labels)
        np.testing.assert_allclose(f['frames/0000/intensity_mean'],[10,100])
        assert f.attrs['resplit_version']=='background_zero_v2'
        assert f['frames/0000'].attrs['complete']
    with pytest.raises(FileExistsError):module.resplit(original,original,image_dir,output,k=0)
