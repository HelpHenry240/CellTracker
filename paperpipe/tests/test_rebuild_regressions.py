"""论文公式、阶段契约和关键失败模式的回归；只使用小型合成数据。"""
import json
import importlib.util
from pathlib import Path
from dataclasses import replace
import h5py
import numpy as np
import pytest
import torch

from papertrack.config import PipelineConfig,MeasureConfig,GraphConfig,TrackletConfig,override
from papertrack.coupling import solve_coupling
from papertrack.coupling.pairwise import PairCoupling
from papertrack.graph.build import build_graph,load_encoder_features,_candidates
from papertrack.temporal.multiscale import product_regularizer,refine_couplings
from papertrack.representation.measure import knn_structure
from papertrack.runtime.ablation import module_names,variant
from papertrack.runtime.contracts import pipeline_contract,check_contract
from papertrack.runtime import run_pipeline
from papertrack.longrange.tracklet import cut_pieces,link_tracklets,_Piece,_candidate_costs,_component_ot
from papertrack.reconstruction.rules import reconstruct_from_edges,mass_flags
from papertrack.reconstruction.exporter import _stamp_hole,export_ctc
from papertrack.runtime.validate import validate_ctc_dir
from papertrack.gnn.train import GraphDataset,TrainConfig,train
from celltracker.track.base import Detections,TrackResult
from celltracker.data.ctc import Track
from celltracker.ot.fgw import regularized_objective
from celltracker.ot.sinkhorn import sinkhorn_log


def test_knn_excludes_self_when_centroids_coincide():
    xy = np.array([[0., 0., 0.], [0., 0., 0.], [2., 0., 0.]])
    _, weights, adjacency = knn_structure(xy, np.ones(3), MeasureConfig(knn_k=1))
    assert not np.diag(adjacency).any()
    assert adjacency[0, 1] and adjacency[1, 0]
    assert weights[0, 1] == 1 and np.isfinite(weights).all()


def test_legacy_knn_empty_singleton_and_coincident_nodes():
    from celltracker.cost.features import gaussian_knn_graph
    for count in (0, 1, 2):
        for neighbors in (0, 1, 6):
            distance, weights = gaussian_knn_graph(np.zeros((count, 3)), k=neighbors)
            assert distance.shape == weights.shape == (count, count)
            assert np.isfinite(distance).all() and np.isfinite(weights).all()
            assert not np.diag(weights).any()
            if count == 2 and neighbors > 0:
                assert weights[0, 1] == weights[1, 0] == 1


def detections(n_frames=8):
    return Detections({t:{'centroid':np.array([[3.,3+t*.1,3],[3,8+t*.1,8]]),
        'volume':np.array([8.,8.]),'label':np.array([1,2]),'gt_label':np.array([1,2]),
        'gt_ids':np.array([[1],[2]]),'intensity_mean':np.array([80.,160.]),
        'intensity_std':np.array([10.,20.])} for t in range(n_frames)},meta={'shape':np.array([10,16,16])})


def config():
    cfg = PipelineConfig()
    cfg.measure.spacing_zyx=(1.,1.,1.)
    cfg.coupling.eta=.1
    cfg.coupling.eps=.5
    cfg.coupling.fgw_outer=3
    cfg.coupling.sinkhorn_iters=100
    cfg.motion.alpha_pred=.2
    cfg.multiscale.ks=(2,)
    cfg.multiscale.lambda_temp={2:.1}
    cfg.multiscale.n_rounds=1
    cfg.gnn.hidden=8
    cfg.gnn.layers=2
    cfg.gnn.lambda_ot=.01
    cfg.detection_source='synthetic'
    return cfg


def write_h5(path,dets):
    with h5py.File(path,'w') as f:
        f.attrs['shape']=dets.meta['shape'].astype(int)
        f.attrs['spacing_zyx']=(1.,1.,1.)
        f.attrs['detection_source']='synthetic'
        f.create_dataset('tracks',data=np.array([(1,0,len(dets.t_range)-1,0),(2,0,len(dets.t_range)-1,0)],
                         dtype=[('label','i4'),('begin','i4'),('end','i4'),('parent','i4')]))
        frames=f.create_group('frames')
        for t,table in dets.frames.items():
            group=frames.create_group(f'{t:04d}')
            for key,value in table.items():
                group.create_dataset(key,data=value)
            mask=np.zeros((10,16,16),dtype='uint16')
            for i,centroid in enumerate(table['centroid']):
                z,y,x=np.round(centroid).astype(int)
                mask[z:z+2,y:y+2,x:x+2]=i+1
            group.create_dataset('labels',data=mask)
    return path


def test_eq6_has_no_extra_factor_two_and_physical_volume():
    D,W,_=knn_structure(np.array([[0.,0,0],[0,1,0]]),np.array([1.,1.]),
        MeasureConfig(knn_k=1,sigma_x=2,sigma_f=2),spacing=(1,2,1),d_full=True)
    assert D[0,1]==2
    assert W[0,1]==pytest.approx(np.exp(-2))


@pytest.mark.parametrize('mode',['raw','cond'])
def test_time_product_gradients_rectangular(mode):
    rng=np.random.default_rng(33)
    plans=[rng.uniform(.1,.5,(2,3)),rng.uniform(.1,.5,(3,4)),rng.uniform(.1,.5,(4,2))]
    direct=rng.uniform(.1,.5,(2,2))
    value,grads=product_regularizer(plans,direct,mode)
    assert value>=0
    for index,plan in enumerate(plans):
        for row,col in [(0,0),(plan.shape[0]-1,plan.shape[1]-1)]:
            plus=[p.copy() for p in plans]; minus=[p.copy() for p in plans]
            plus[index][row,col]+=1e-6; minus[index][row,col]-=1e-6
            numerical=(product_regularizer(plus,direct,mode)[0]-product_regularizer(minus,direct,mode)[0])/2e-6
            assert grads[index][row,col]==pytest.approx(numerical,rel=1e-5,abs=1e-7)


def test_complete_objective_includes_entropy_kl_and_forbidden_edges():
    P=np.array([[.2,.1],[.1,.2]])
    C=np.array([[1.,2.],[2.,1.]])
    a=b=np.array([.5,.5])
    row=P.sum(1)
    expected=np.sum(P*C)+.4*np.sum(P*(np.log(P)-1))+3*np.sum(row*np.log(row/a)-row+a)
    assert regularized_objective(P,C,None,None,a,b,0,.4,1,2)==pytest.approx(expected)
    C[0,0]=np.inf
    assert np.isinf(regularized_objective(P,C,None,None,a,b,0,.4,1,2))


def test_multiscale_decreases_complete_objective_without_replacing_cost():
    d=detections(4); cfg=config()
    coupling={t:solve_coupling(d.centroid(t),d.centroid(t+1),d.volume(t),d.volume(t+1),
            coupling_cfg=cfg.coupling,measure_cfg=cfg.measure,spacing=(1,1,1)) for t in range(3)}
    costs={t:art.cost.copy() for t,art in coupling.items()}
    result=refine_couplings(coupling,d.t_range,{t:d.centroid(t) for t in d.t_range},
              {t:d.volume(t) for t in d.t_range},replace(cfg.multiscale,n_rounds=3),cfg.coupling,cfg.measure,(1,1,1))
    history=[entry['objective'] for entry in result.history]
    assert np.isfinite(history).all()
    assert np.all(np.diff(history)<=1e-10)
    for t,art in result.couplings.items():
        np.testing.assert_array_equal(art.cost,costs[t])
        np.testing.assert_array_equal(coupling[t].cost,costs[t])


def test_context_contains_connected_historical_time_edges():
    d=detections(5); cfg=config()
    coupling={t:solve_coupling(d.centroid(t),d.centroid(t+1),d.volume(t),d.volume(t+1),
                coupling_cfg=cfg.coupling,measure_cfg=cfg.measure,spacing=(1,1,1)) for t in range(4)}
    graph=build_graph(d,2,d.t_range,coupling[2],None,replace(cfg.graph,bridge=False,ctx_window=2),
                      cfg.measure,spacing=(1,1,1),context_couplings=coupling)
    src,dst=graph['edge_index']; nf=graph['node_frame']
    time_pairs=set(zip(nf[src].tolist(),nf[dst].tolist()))
    assert {(0,1),(1,2),(2,3)}<=time_pairs
    assert np.all(nf[src[graph['is_target']]]==2)


@pytest.mark.parametrize('identities',['label','centroid'])
def test_encoder_alignment_reorders_equal_count(tmp_path,identities):
    d=detections(1)
    values={'frame_0':np.array([[20.,21.],[10.,11.]])}
    values[f'{identities}_0']=d.label(0)[::-1] if identities=='label' else d.centroid(0)[::-1]
    np.savez(tmp_path/'encoder.npz',**values)
    features=load_encoder_features(tmp_path/'encoder.npz',d,(1,1,1))
    np.testing.assert_array_equal(features[0],[[10,11],[20,21]])


def test_encoder_missing_frame_is_error(tmp_path):
    np.savez(tmp_path/'encoder.npz',frame_0=np.zeros((2,4)),label_0=[1,2])
    with pytest.raises(ValueError,match='缺帧'):
        load_encoder_features(tmp_path/'encoder.npz',detections(2))


def test_topk_zero_disables_fallback():
    art=PairCoupling(np.array([[.01,.5]]),np.array([[.1,.2]]),np.array([1.]),np.array([.5,.5]),np.ones((1,2)),.1)
    cfg=GraphConfig(theta_gamma=.1,cand_topk=0)
    assert _candidates(art,cfg,3)[0].tolist()==[[0,1]]
    assert _candidates(art,replace(cfg,cand_topk=2),3)[0].tolist()==[[0,0],[0,1]]


def test_undersegmentation_supervision_retains_all_gt_identities():
    d=detections(2)
    d.frames[0]['gt_ids']=np.array([[1,2],[0,0]])
    d.frames[0]['gt_label']=np.array([1,0])
    art=PairCoupling(np.ones((2,2))*.1,np.zeros((2,2)),np.ones(2)*.5,np.ones(2)*.5,np.zeros((2,2)),.1)
    graph=build_graph(d,0,d.t_range,art,None,GraphConfig(bridge=False),MeasureConfig(),gt_parent={})
    assert graph['label'].tolist()==[1,1,0,0]
    assert graph['cand_label_div'].sum()==0


def test_absolute_birth_death_flags_and_veto_switch():
    d=detections(2); cfg=config().reconstruct
    art=PairCoupling(np.eye(2)*.1,np.zeros((2,2)),np.ones(2)*.5,np.ones(2)*.5,np.zeros((2,2)),.1)
    cfg=replace(cfg,eta_death=.15,eta_birth=0)
    assert mass_flags(art,cfg)[0].all()
    assert not mass_flags(art,replace(cfg,mass_threshold_mode='relative'))[0].any()
    decisions={0:{'pairs':np.array([[0,0]]),'score':np.array([.9])}}
    assert reconstruct_from_edges(d,decisions,cfg,{0:art}).meta['n_mass_veto']==1
    assert reconstruct_from_edges(d,decisions,replace(cfg,birth_death_enabled=False),{0:art}).meta['n_edges_accepted']==1


def test_division_volume_and_uncertainty_switches():
    d=detections(2); d.frames[0]['volume'][0]=16
    decisions={0:{'pairs':np.array([[0,0],[0,1],[1,0]]),'score':np.array([.95,.9,.8])}}
    cfg=config().reconstruct
    result=reconstruct_from_edges(d,decisions,cfg)
    assert result.meta['n_division']==1
    assert result.meta['n_uncertain_nodes']>=1
    assert reconstruct_from_edges(d,decisions,replace(cfg,division_enabled=False)).meta['n_division']==0
    d.frames[0]['volume'][0]=100
    assert reconstruct_from_edges(d,decisions,cfg).meta['n_division']==0
    assert reconstruct_from_edges(d,decisions,replace(cfg,volume_conservation=False)).meta['n_division']==1


def test_isolated_filter_keeps_background_out_of_tracks():
    d=detections(2)
    result=reconstruct_from_edges(d,{},replace(config().reconstruct,filter_isolated=True))
    assert result.n_tracks()==0
    assert all(not a.any() for a in result.assignment.values())
    pieces,_=cut_pieces(result,d,3)
    assert pieces==[]


def test_sliding_tracklet_uses_confidence_and_average_velocity():
    d=detections(4)
    result=TrackResult({t:np.array([1,2]) for t in range(4)},
        {1:Track(1,0,3,0),2:Track(2,0,3,0)},meta={'accepted_edges':[
            [t,i,t+1,i,.6 if t==1 and i==0 else .95] for t in range(3) for i in range(2)]})
    pieces,_=cut_pieces(result,d,3,.8)
    assert [p.frames for p in pieces if p.orig==1]==[[0,1],[2,3]]
    assert [p.frames for p in pieces if p.orig==2]==[[0,1,2,3]]
    np.testing.assert_allclose(pieces[0].velocity,[0,.1,0])


def test_tracklet_distances_use_spacing_and_source_has_only_one_successor():
    A=_Piece(0,1,[0],[np.array([0.,0,0])],[8.])
    B=_Piece(1,2,[1],[np.array([0.,0,10])],[8.])
    cfg=TrackletConfig(max_gap=1)
    d=detections(2); ccfg=config().coupling
    pairs,_=_candidate_costs([A,B],d,cfg,replace(ccfg,r_max=2),(1,1,.1))
    assert pairs.tolist()==[[0,1]]
    pairs,_=_candidate_costs([A,B],d,cfg,replace(ccfg,r_max=2),(1,1,1))
    assert pairs.size==0


def test_tracklet_component_solver_equals_dense_masked_ot():
    pairs=np.array([[0,1],[2,3]])
    costs=np.array([.2,.4]); mass=np.ones(4)*.25; cfg=TrackletConfig(tau=.5)
    flows,components,entries=_component_ot(pairs,costs,mass,.1,cfg,1000)
    # 同一全局质量求两个独立 1×1 问题，无额外重新归一化。
    expected=np.array([sinkhorn_log(np.array([[costs[k]]]),mass[:1],mass[:1],eps=.1,
                            tau_a=.5,tau_b=.5)[0,0] for k in range(2)])
    np.testing.assert_allclose(flows,expected)
    assert components==2 and entries==1


def test_tracklet_parent_points_to_terminal_piece():
    d=detections(4)
    result=TrackResult({0:np.array([1,0]),1:np.array([1,0]),2:np.array([2,3]),3:np.array([2,3])},
        {1:Track(1,0,1,0),2:Track(2,2,3,1),3:Track(3,2,3,1)},
        {'accepted_edges':[[0,0,1,0,.6],[2,0,3,0,.95],[2,1,3,1,.95]]})
    cfg=replace(config().tracklet,theta_link=1e6,max_gap=1)
    result=link_tracklets(d,result,cfg,config().coupling,config().measure,(1,1,1)).result
    terminal=int(result.assignment[1][0])
    assert terminal!=int(result.assignment[0][0])
    assert result.tracks[int(result.assignment[2][0])].parent==terminal
    assert result.tracks[int(result.assignment[2][1])].parent==terminal


def test_hole_stamp_never_overwrites_existing_instance():
    d=detections(3)
    labels=np.zeros((10,16,16),dtype='uint16'); labels[3:5,3:5,3:5]=1
    res=np.full_like(labels,2)
    status=_stamp_hole(res,{'frames/0000/labels':labels},d,{0:np.array([1,2]),2:np.array([1,2])},1,0,2,1)
    assert status=='failed'
    assert np.all(res==2)


def test_contract_is_sequence_path_independent_and_configuration_strict():
    cfg=config(); cfg.node.f_source='encoder_npz'; cfg.node.encoder_feat_path='/a/seq01.npz'
    encoder={'schema':'encoder_v1','model_sha256':'model','feature_dim':128,'stage':2}
    first=pipeline_contract(cfg,{'spacing_zyx':(1,.09,.09)},encoder)
    second=pipeline_contract(override(cfg,['node.encoder_feat_path=/b/seq02.npz']),{'spacing_zyx':(1,.09,.09)},encoder)
    check_contract(first,second)
    with pytest.raises(ValueError):
        check_contract(first,pipeline_contract(override(cfg,['graph.bridge=false']),{},encoder))


def test_frontend_contract_rejects_resplit_and_seed_recipe_mismatch():
    cfg=config();cfg.detection_source='nnunet_pred'
    meta={'spacing_zyx':(1,.09,.09),'h_frac':np.float64(.1),'min_volume':np.int64(300),
          'gaussian_sigma':0.,'watershed':True,'oracle_markers':False,'resplit_k':0.,'min_distance':3}
    original=pipeline_contract(cfg,meta)
    check_contract(original,pipeline_contract(cfg,{**meta,'seq':'02','min_distance':8}))
    # h-maxima 不使用 min_distance；改变再切规则却必须拒绝旧权重。
    with pytest.raises(ValueError):check_contract(original,pipeline_contract(cfg,{**meta,'resplit_k':1.6}))
    with pytest.raises(ValueError,match='前端来源参数'):pipeline_contract(cfg,{'spacing_zyx':(1,.09,.09)})


def test_detection_loader_keeps_frontend_provenance(tmp_path):
    path=write_h5(tmp_path/'detections.h5',detections(3))
    with h5py.File(path,'r+') as f:
        f.attrs.update(h_frac=.1,min_volume=300,gaussian_sigma=0.,watershed=True,
                       oracle_markers=False,resplit_k=1.6,resplit_version='background_zero_v2')
    dets=Detections.from_h5(path)
    assert dets.meta['resplit_k']==1.6 and dets.meta['min_volume']==300
    assert dets.meta['resplit_version']=='background_zero_v2'


def test_ablation_defaults_flip_and_ot_consumers_are_removed():
    cfg=config(); cfg.node.f_source='encoder_npz'
    for name in module_names():
        varied,report=variant(cfg,name)
        assert varied!=cfg
        assert report['changes']
    assert variant(cfg,'topk')[1]['direction']=='on'
    assert variant(cfg,'fgw')[1]['direction']=='off'
    no_ot,_=variant(cfg,'ot')
    assert not no_ot.coupling.enabled and not no_ot.graph.use_ot_features and no_ot.gnn.lambda_ot==0
    assert cfg.coupling.enabled


def test_all_modules_train_infer_export_and_resume(tmp_path):
    d=detections(10); h5=write_h5(tmp_path/'sequence.h5',d); cfg=config()
    sidecar={f'frame_{t}':np.array([[.1,.2],[.8,.9]],dtype='float32') for t in d.t_range}
    sidecar.update({f'label_{t}':d.label(t) for t in d.t_range})
    sidecar['metadata_json']=np.array(json.dumps({'schema':'synthetic_encoder','model_sha256':'fixed','feature_dim':2,'stage':0}))
    np.savez(tmp_path/'encoder.npz',**sidecar)
    cfg.node.f_source='encoder_npz'; cfg.node.encoder_feat_path=str(tmp_path/'encoder.npz')
    built=run_pipeline(h5,cfg,gt_h5=h5,dump_graphs=tmp_path/'graphs',cache_dir=tmp_path/'cache',verbose=False)
    assert built.info['multiscale']['enabled'] and built.info['motion']['enabled']
    assert built.info['tracklet']['enabled'] and built.info['f_source']=='encoder_npz'
    train_set=GraphDataset(tmp_path/'graphs','train',val_fraction=.3)
    val_set=GraphDataset(tmp_path/'graphs','val',val_fraction=.3)
    assert len(train_set)>0
    train_end=max(int(np.load(path)['support_end']) for path in train_set.sel)
    val_begin=min(int(np.load(path)['support_start']) for path in val_set.sel)
    assert train_end<val_begin
    training=TrainConfig(epochs=2,hidden=8,layers=2,batch_pairs=2,lambda_ot=.01,val_fraction=0,
                        out_dir=str(tmp_path/'model'),seed=6)
    result=train(tmp_path/'graphs',training)
    assert result['history'][-1]['graphs_seen']==9
    resumed=train(tmp_path/'graphs',replace(training,epochs=3,resume=True))
    assert resumed['history'][-1]['epoch']==3
    uninterrupted=train(tmp_path/'graphs',replace(training,epochs=3,out_dir=str(tmp_path/'full_model')))
    checkpoint=torch.load(tmp_path/'model/last.pt',weights_only=False)
    other=torch.load(tmp_path/'full_model/last.pt',weights_only=False)
    for key,value in checkpoint['model'].items():
        torch.testing.assert_close(value,other['model'][key],rtol=0,atol=0)
    predicted=run_pipeline(h5,cfg,ckpt=tmp_path/'model/best.pt',cache_dir=tmp_path/'cache',verbose=False)
    stats=export_ctc(predicted,h5,tmp_path/'01_RES',gt_h5=h5)
    assert validate_ctc_dir(tmp_path/'01_RES',expected_frames=d.t_range)['ok']
    assert stats['hole_frames_overwritten']==0
    with pytest.raises(ValueError,match='缓存输入/配置'):
        run_pipeline(h5,override(cfg,['coupling.beta=0']),cache_dir=tmp_path/'cache',verbose=False)
    with pytest.raises(ValueError,match='图配置不一致'):
        run_pipeline(h5,override(cfg,['graph.intra_enabled=false']),ckpt=tmp_path/'model/best.pt',verbose=False)
    # GT 身份只用于监督与诊断，推理不能因改写 GT 身份而改变预测。
    with h5py.File(h5,'r+') as f:
        for group in f['frames'].values():
            group['gt_label'][:]=0
            group['gt_ids'][:]=0
    without_truth=run_pipeline(h5,cfg,ckpt=tmp_path/'model/best.pt',verbose=False)
    for t in d.t_range:
        np.testing.assert_array_equal(predicted.result.assignment[t],without_truth.result.assignment[t])
    assert predicted.result.tracks==without_truth.result.tracks


def test_encoder_coordinate_mapping_and_tiled_sampling():
    path=Path(__file__).resolve().parents[2]/'scripts/nnunet/export_encoder_features.py'
    spec=importlib.util.spec_from_file_location('encoder_export',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    mapped=module.transform_centroids(np.array([[4.,8.,12.]]),[2,1,0],[[2,20],[0,20],[1,10]],[18,20,9],[9,10,9])
    np.testing.assert_allclose(mapped,[[4.75,3.75,3.]])
    data=np.ones((1,4,4,4),dtype='float32')*7
    class Encoder(torch.nn.Module):
        def forward(self,x):
            return [x]
    features=module.extract_frame(Encoder(),data,np.array([[0.,0,0],[3.,3,3]]),(3,3,3),0,torch.device('cpu'))
    np.testing.assert_allclose(features,[[7],[7]])


def test_calibration_reports_raw_mass_and_candidate_loss(tmp_path):
    from papertrack.runtime.calibration import calibrate_run
    d=detections(4); path=write_h5(tmp_path/'data.h5',d)
    run=run_pipeline(path,config(),verbose=False)
    cfg,report=calibrate_run(run,{1:0,2:0})
    assert cfg.graph.theta_gamma is not None and cfg.graph.theta_gamma_frac is None
    assert cfg.reconstruct.mass_threshold_mode=='absolute'
    assert report['counts']['true_detection_pairs']==6
    assert 0<=report['after_calibration']['candidate_true_recall']<=1
    assert report['physical_volume_um3']['p50']==8


def test_official_adapter_rejects_missing_metric_and_preserves_logs(tmp_path,monkeypatch):
    from papertrack.runtime.official import evaluate_official
    import subprocess
    d=detections(3); path=write_h5(tmp_path/'data.h5',d)
    run=run_pipeline(path,config(),verbose=False)
    export_ctc(run,path,tmp_path/'01_RES')
    tools=tmp_path/'tools'; tools.mkdir(); gt=tmp_path/'01_GT'; gt.mkdir()
    for name in ['DET','SEG','TRA']:
        (tools/f'{name}Measure').write_text(name)
    calls=[]
    def runner(command,**kwargs):
        name=Path(command[0]).name[:3]; calls.append(name)
        return subprocess.CompletedProcess(command,0,stdout=f'{name} measure: 0.9\n',stderr='')
    monkeypatch.setattr(subprocess,'run',runner)
    result=evaluate_official(tmp_path/'01_RES',gt,tools,'01',tmp_path/'official.json')
    assert result['DET']==.9 and calls==['DET','SEG','TRA']
    assert len(list((tmp_path/'official_logs').glob('*stdout.txt')))==3
    monkeypatch.setattr(subprocess,'run',lambda command,**kw:subprocess.CompletedProcess(command,0,'no metric',''))
    with pytest.raises(RuntimeError,match='官方评测失败'):
        evaluate_official(tmp_path/'01_RES',gt,tools,'01',tmp_path/'failed_official.json')
    assert not (tmp_path/'failed_official.json').exists()


def test_resume_graphs_checks_contents_before_reuse(tmp_path):
    d=detections(4); path=write_h5(tmp_path/'data.h5',d); cfg=config()
    run_pipeline(path,cfg,gt_h5=path,dump_graphs=tmp_path/'graphs',verbose=False)
    stored=tmp_path/'graphs/pair_0000.npz'; original=stored.read_bytes()
    run_pipeline(path,cfg,gt_h5=path,dump_graphs=tmp_path/'graphs',resume_graphs=True,verbose=False)
    assert stored.read_bytes()==original
    with np.load(stored) as data:
        changed={k:data[k] for k in data.files}
    changed['label']=1-changed['label']; np.savez_compressed(stored,**changed)
    with pytest.raises(ValueError,match='已有图与恢复配置'):
        run_pipeline(path,cfg,gt_h5=path,dump_graphs=tmp_path/'graphs',resume_graphs=True,verbose=False)


def test_matrix_plan_has_independent_models_and_seq02_holdout(tmp_path):
    import argparse
    from papertrack.config import save_config
    path=Path(__file__).resolve().parents[1]/'scripts/run_ablation_matrix.py'
    spec=importlib.util.spec_from_file_location('matrix',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    save_config(config(),tmp_path/'config.yaml')
    args=argparse.Namespace(config=str(tmp_path/'config.yaml'),out=str(tmp_path/'plan'),
        ablations=['fgw','gnn'],encoder01=None,encoder02=None,h5_01='pred01.h5',h5_02='pred02.h5',
        gt_01='gt01.h5',gt_02='gt02.h5',exp_prefix='smoke',device='cpu',frames='0:8',
        seeds=[1,2],epochs=2,official=False,dataset='synthetic')
    plan=module.make_plan(args)
    trained=[job for job in plan['jobs'] if job['kind']=='train']
    assert len(trained)==4 and len({job['outputs'][0] for job in trained})==4
    for job in trained:
        assert job['command'][job['command'].index('--graphs')+1].endswith(('baseline','fgw'))
    evals=[job for job in plan['jobs'] if job['kind']=='evaluate']
    assert len(evals)==12
    assert all('--ckpt' not in job['command'] for job in evals if '_gnn_' in job['id'])


def test_matrix_recalibrates_on_training_sequence_and_selects_test_encoder(tmp_path):
    import argparse
    from papertrack.config import save_config
    path=Path(__file__).resolve().parents[1]/'scripts/run_ablation_matrix.py'
    spec=importlib.util.spec_from_file_location('matrix_calibration',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cfg=config(); cfg.node.f_source='encoder_npz'; cfg.node.encoder_feat_path='encoder01.npz'
    save_config(cfg,tmp_path/'config.yaml')
    args=argparse.Namespace(config=str(tmp_path/'config.yaml'),out=str(tmp_path/'plan'),
        ablations=['fgw'],encoder01='encoder01.npz',encoder02='encoder02.npz',h5_01='pred01.h5',h5_02='pred02.h5',
        gt_01='gt01.h5',gt_02='gt02.h5',exp_prefix='smoke',device='cpu',frames='0:8',
        seeds=[1,2],epochs=2,official=False,dataset='synthetic',recalibrate=True,skip_baseline=True)
    plan=module.make_plan(args)
    calibration=[job for job in plan['jobs'] if job['kind']=='calibrate']
    assert len(calibration)==1 and 'gt01.h5' in calibration[0]['command']
    assert 'gt02.h5' not in calibration[0]['command']
    build=next(job for job in plan['jobs'] if job['kind']=='build')
    assert build['depends']==[calibration[0]['id']]
    eval02=next(job for job in plan['jobs'] if job['id'].endswith('eval02'))
    assert 'node.encoder_feat_path=encoder02.npz' in eval02['command']
    # 基准也走相同标定步骤，避免把“重新标定”的收益归到被关闭的模块。
    args.out=str(tmp_path/'with_baseline'); args.skip_baseline=False
    controlled=module.make_plan(args)
    calibration=[job for job in controlled['jobs'] if job['kind']=='calibrate']
    assert len(calibration)==2
    assert any(job['id']=='smoke_baseline_calibration01' for job in calibration)


def test_compressed_ctc_export_is_pixel_identical(tmp_path):
    import tifffile
    from celltracker.eval.ctc_io import ResultWriter
    labels=np.zeros((4,24,24),dtype=np.uint16); labels[1:3,5:12,6:14]=65534
    writer=ResultWriter(tmp_path,compression='zlib')
    path=writer.add(3,labels)
    assert np.array_equal(tifffile.imread(path),labels)
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].photometric==1  # 3/4 张切片也必须是标量标签栈，不能被当成 RGB。
    assert path.stat().st_size<labels.nbytes
