"""Read-only numerical probes for ideas.pdf audit; no training or cloud calls."""
import argparse
import copy
import json
import sys
import subprocess
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default='/home/henry/ot_idea/CellTracker')
ap.add_argument('--out', default='/tmp/ideas_pipeline_probe_results.json')
args = ap.parse_args()
sys.path.insert(0, str(Path(args.repo) / 'paperpipe/src'))
import papertrack
import numpy as np
import torch
from celltracker.track.base import Detections, TrackResult
from celltracker.data.ctc import Track
from celltracker.ot.fgw import fused_gw, structural_term
from celltracker.ot.sinkhorn import sinkhorn_log
from papertrack.config import (MeasureConfig, CouplingConfig, GraphConfig,
                              ReconstructConfig, MultiscaleConfig, TrackletConfig)
from papertrack.coupling.pairwise import PairCoupling, solve_coupling
from papertrack.representation.measure import knn_structure
from papertrack.temporal.multiscale import entropy, refine_couplings
from papertrack.graph.build import build_graph, load_encoder_features
from papertrack.gnn.model import EdgeGNN, ModelConfig
from papertrack.gnn.infer import predict_graph
from papertrack.reconstruction.rules import reconstruct_from_ot, reconstruct_from_edges
from papertrack.longrange.tracklet import link_tracklets

torch.set_num_threads(1)
results = {}
def dets(cents, vols=None):
    frames = {}
    for t, xy in enumerate(cents):
        xy = np.asarray(xy, dtype=float).reshape(-1, 3)
        n = len(xy)
        frames[t] = dict(centroid=xy, volume=np.full(n, 100.0) if vols is None else np.asarray(vols[t]),
                         label=np.arange(1,n+1), gt_label=np.arange(1,n+1),
                         intensity_mean=np.full(n,100.), intensity_std=np.full(n,10.))
    return Detections(frames, meta={'shape': np.array([32,64,64])})

# Eq (11), including the non-balanced case where total mass is variable.
P = np.array([[.2,.1],[.1,.1]])
expected = float(np.sum(P*(np.log(P)-1)))
results['entropy_eq11'] = dict(expected=expected, observed=entropy(P),
                              error=entropy(P)-expected, total_mass=float(P.sum()))

# Historical nodes exist, but no historical temporal edges connect them to target nodes.
d = dets([[[0,0,t],[0,10,t]] for t in range(4)])
cc = CouplingConfig(eta=0, beta=0, r_max=20, eps=.5, eps_rel=None, tau_a=None, tau_b=None)
mc = MeasureConfig(knn_k=1)
c = solve_coupling(d.centroid(1),d.centroid(2),d.volume(1),d.volume(2),coupling_cfg=cc,measure_cfg=mc)
g = build_graph(d,1,d.t_range,c,None,GraphConfig(ctx_window=1,bridge=False,theta_gamma_frac=0),mc)
edges = g['edge_index']
fr = g['node_frame']
frame_pairs = sorted(set(zip(fr[edges[0]].tolist(),fr[edges[1]].tolist())))
torch.manual_seed(7)
model = EdgeGNN(ModelConfig(hidden=8,layers=3,dropout=0)).eval()
g2 = copy.deepcopy(g)
g2['node_feat'][fr==0] += 1000
p1,p2 = predict_graph(model,g)['score'],predict_graph(model,g2)['score']
results['temporal_context'] = dict(frames=sorted(set(fr.tolist())),edge_frame_pairs=frame_pairs,
    historical_perturbation_max_probability_change=float(np.max(np.abs(p1-p2))))

# Empty ancestry is valid on sequences with no divisions; None means unavailable supervision.
label_none=build_graph(d,1,d.t_range,c,None,GraphConfig(ctx_window=1,bridge=False,theta_gamma_frac=0),mc,gt_parent=None)
label_empty=build_graph(d,1,d.t_range,c,None,GraphConfig(ctx_window=1,bridge=False,theta_gamma_frac=0),mc,gt_parent={})
results['empty_ancestry_labels'] = dict(runtime_argument='gt_parent or None',
    positive_labels_with_none=int(label_none['label'].sum()),
    positive_labels_with_valid_empty_dict=int(label_empty['label'].sum()))

# A sidecar with equal cardinality but permuted centroids must still be aligned.
with tempfile.TemporaryDirectory(prefix='ideas-encoder-audit-') as temp:
    fp=Path(temp)/'encoder.npz'
    np.savez(fp,frame_0=np.array([[20.],[10.]]),centroid_0=d.centroid(0)[::-1])
    results['encoder_alignment'] = dict(expected=[10.,20.],
        observed=load_encoder_features(fp,d)[0][:,0].tolist())

# Show the new trainer's per-batch reshuffle repeats/omits examples inside one epoch.
rng_order=np.random.RandomState(7)
order=np.concatenate([rng_order.permutation(10)[start:start+2] for start in range(0,10,2)])
results['training_epoch_coverage'] = dict(n_graphs=10,n_draws=len(order),
    n_unique=len(set(order.tolist())),order=order.tolist())

# Actual death_veto field does not change reconstruction, even when every row is below death threshold.
small = dets([[[0,0,0],[0,10,0]],[[0,0,1],[0,10,1]]])
art = PairCoupling(plan=np.eye(2)*.05,cost=np.array([[1.,np.inf],[np.inf,1.]]),
                  mass_a=np.full(2,.5),mass_b=np.full(2,.5),d_cur=np.eye(2),eps_eff=.1)
death = {}
for flag in [False,True]:
    rc = ReconstructConfig(theta_gamma_frac=.01,eta_death=.9,eta_birth=.9,death_veto=flag)
    res = reconstruct_from_ot(small,{0:art},rc,3,bridge=False)
    death[str(flag)] = dict(assignment={str(t):a.tolist() for t,a in res.assignment.items()},
                           n_death=res.meta['n_death'],n_birth=res.meta['n_birth'])
results['death_veto'] = death

# GNN edge suppression does not remove isolated false-positive detections from result.
iso = reconstruct_from_edges(small,{},ReconstructConfig())
results['isolated_nodes'] = dict(input_nodes=4,output_tracks=iso.n_tracks(),
                                assignment={str(t):a.tolist() for t,a in iso.assignment.items()})

# Tracklet units: 10 xy voxels = 0.9 um, inside 3 um, yet the voxel gate rejects the pair.
td = dets([[[0,0,0]],[[0,0,10]]])
tr = TrackResult(assignment={0:np.array([1]),1:np.array([2])},
                 tracks={1:Track(1,0,0,0),2:Track(2,1,1,0)})
tl = link_tracklets(td,tr,TrackletConfig(window=5,max_gap=1,theta_link=0),
                    CouplingConfig(eta=0,beta=0,r_max=3,eps=.1,eps_rel=None),mc,(1,.09,.09))
results['tracklet_units'] = dict(displacement_voxel=10,displacement_um=.9,r_max_um=3,
                               actual_candidates=tl.info['n_candidates'],actual_links=tl.info['n_links'])

# Verify that a single-node frame is supported (a suspected issue that did not reproduce).
try:
    knn_structure(np.zeros((1,3)),np.ones(1),mc)
    results['singleton_knn'] = {'error':None}
except Exception as exc:
    results['singleton_knn'] = {'error':f'{type(exc).__name__}: {exc}'}

# Eq (6) has sigma^2, not 2*sigma^2. Here both feature and spatial differences are exactly 1.
_,W,_ = knn_structure(np.array([[0,0,0],[0,0,1]]),np.array([1.,1.]),
                     MeasureConfig(knn_k=1,sigma_x=1,sigma_f=1))
results['weight_eq6'] = dict(expected=float(np.exp(-2)),observed=float(W[0,1]))

# A split parent must map to its TERMINAL piece, not its first piece.
lineage_dets = dets([[[0,0,t]] for t in range(6)] + [[[0,0,6],[0,1,6]]])
lineage_input = TrackResult(assignment={**{t:np.array([1]) for t in range(6)},6:np.array([2,3])},
    tracks={1:Track(1,0,5,0),2:Track(2,6,6,1),3:Track(3,6,6,1)})
lineage_out = link_tracklets(lineage_dets,lineage_input,
    TrackletConfig(window=3,max_gap=1,theta_link=1e9),
    CouplingConfig(eta=0,beta=0,r_max=100,eps=.1,eps_rel=None),mc)
results['tracklet_parent_mapping'] = dict(input_children_with_parent=2,
    output_children_with_parent=sum(tr.parent>0 for tr in lineage_out.result.tracks.values()),
    n_links=lineage_out.info['n_links'],parent_fixed=lineage_out.info.get('parent_fixed'),
    tracks={str(k):dict(begin=tr.begin,end=tr.end,parent=tr.parent) for k,tr in lineage_out.result.tracks.items()})

# Scalar eq (14) has an analytic minimum: p=exp(-(1-eta)C/(eps+tau_a+tau_b)).
scalar_p,scalar_info=fused_gw(np.array([[3.]]),np.ones(1),np.ones(1),
    np.zeros((1,1)),np.zeros((1,1)),eta=.5,eps=1.,tau_a=1.,tau_b=1.)
results['fgw_scalar_analytic'] = dict(expected=float(np.exp(-.5)),observed=float(scalar_p[0,0]),
                                     outer_iters=scalar_info['iters'])

# Training label mapping must divide overlap by the COMPLETE GT marker volume.
with tempfile.TemporaryDirectory(prefix='ideas-label-audit-') as temp:
    import SimpleITK as sitk
    import tifffile
    import h5py
    base=Path(temp)
    (base/'pred').mkdir();(base/'images/01').mkdir(parents=True)
    (base/'gt/01_GT/TRA').mkdir(parents=True)
    mask=np.zeros((3,3,12),dtype=np.uint8);mask[1,1,1:3]=1
    truth=np.zeros_like(mask,dtype=np.uint16);truth[1,1,1:11]=1
    sitk.WriteImage(sitk.GetImageFromArray(mask),str(base/'pred/CE01_f000.nii.gz'))
    tifffile.imwrite(base/'images/01/t000.tif',np.ones_like(truth))
    tifffile.imwrite(base/'gt/01_GT/TRA/man_track000.tif',truth)
    cmd=[sys.executable,str(Path(args.repo)/'scripts/predict_to_h5.py'),
         '--pred-dir',str(base/'pred'),'--img-root',str(base/'images'),
         '--seq','01','--out',str(base/'out.h5'),'--gt-root',str(base/'gt'),
         '--no-watershed','--min-volume','1','--gaussian-sigma','0']
    completed=subprocess.run(cmd,capture_output=True,text=True,check=True)
    with h5py.File(base/'out.h5','r') as hf:
        mapped=hf['frames/0000/gt_label'][:].tolist()
    results['gt_mapping_denominator'] = dict(overlap_voxels=2,full_marker_voxels=10,
        expected_gt_label=[0],observed_gt_label=mapped)

# Verify that refinement replaces physical feature cost C with its optimisation surrogate G.
for seed in range(12):
    rng=np.random.default_rng(seed)
    xy={t:rng.uniform(0,4,(3,3)) for t in range(4)}
    vol={t:rng.uniform(50,150,3) for t in range(4)}
    cc2=CouplingConfig(eta=0,beta=1,r_max=100,eps=1,eps_rel=None,tau_a=1,tau_b=1,sinkhorn_iters=100)
    cs={t:solve_coupling(xy[t],xy[t+1],vol[t],vol[t+1],coupling_cfg=cc2,measure_cfg=mc) for t in range(3)}
    orig={t:c.cost.copy() for t,c in cs.items()}
    refined=refine_couplings(cs,list(range(4)),xy,vol,
                            MultiscaleConfig(ks=(2,),lambda_temp={2:1.},n_rounds=2),cc2,mc)
    diffs={str(t):float(np.max(np.abs(refined.couplings[t].cost-orig[t]))) for t in cs}
    if max(diffs.values())>1e-9:
        results['refinement_cost_mutation']=dict(seed=seed,max_cost_changes=diffs,history=refined.history,
            min_cost=float(min(c.cost.min() for c in refined.couplings.values())))
        break

# FGW outer acceptance uses only L. Look for returned non-stationary points of the full eq (14).
def full_obj(plan,C,D,Dp,a,b,eta,eps,tau):
    s=plan.sum(1);q=plan.sum(0)
    ent=np.sum(plan[plan>0]*(np.log(plan[plan>0])-1))
    kl=lambda u,v:np.sum(np.where(u>0,u*np.log(np.maximum(u,1e-300)/v),0)-u+v)
    return float((1-eta)*np.sum(C*plan)+eta*structural_term(plan,D,Dp)+eps*ent+tau*(kl(s,a)+kl(q,b)))
for seed in range(60):
    rng=np.random.default_rng(seed)
    C=rng.uniform(0,4,(2,2));D=np.array([[0,rng.uniform(.2,3)],[0,0.]]);D=D+D.T
    Dp=np.array([[0,rng.uniform(.2,3)],[0,0.]]);Dp=Dp+Dp.T
    a=b=np.full(2,.5);eta=.7;eps=1.;tau=1.
    fp,info=fused_gw(C,a,b,D,Dp,eta=eta,eps=eps,tau_a=tau,tau_b=tau,n_outer=10)
    grad=((1-eta)*C+eta*__import__('celltracker.ot.fgw',fromlist=['structural_grad']).structural_grad(fp,D,Dp)
          +eps*np.log(np.maximum(fp,1e-300))+tau*np.log(np.maximum(fp.sum(1),1e-300)/a)[:,None]
          +tau*np.log(np.maximum(fp.sum(0),1e-300)/b)[None,:])
    step=1e-3
    trial=np.maximum(fp-step*grad,1e-12)
    old=full_obj(fp,C,D,Dp,a,b,eta,eps,tau);new=full_obj(trial,C,D,Dp,a,b,eta,eps,tau)
    if new<old-1e-4:
        results['fgw_full_objective']=dict(seed=seed,returned_objective=old,
            gradient_step_objective=new,improvement=old-new,max_gradient=float(np.max(np.abs(grad))),
            outer_iters=info['iters'],L_only_history=info['history'])
        break

Path(args.out).write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
