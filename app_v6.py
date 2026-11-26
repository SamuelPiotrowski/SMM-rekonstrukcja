from pathlib import Path
import itertools, json, threading, uuid, traceback
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from flask import Flask, jsonify, request, render_template, Response
from plotly.offline import get_plotlyjs
from osteo_engine_v6 import metrics, penetration, moved, unit, synth, beam_assemble, refine_contact, optimize_poses

from signature_v6 import describe, candidates as signature_candidates
from local_v6 import proposals as local_proposals

ROOT=Path(__file__).resolve().parent
FULL=ROOT/'02_FRAGMENTY_ZLAMANE'/'Vienna_Brick'/'PLY'
ROI=ROOT/'patches_vienna'
app=Flask(__name__); REG=o3d.pipelines.registration
jobs={}; lock=threading.Lock()

def read(p):
    c=o3d.io.read_point_cloud(str(p)); A=np.asarray(c.points).copy(); N=np.asarray(c.normals).copy()
    if len(A)<30 or not np.isfinite(A).all(): raise ValueError(f'Niepoprawne dane: {p.name}')
    if N.shape!=A.shape or not np.isfinite(N).all(): raise ValueError(f'Brak poprawnych normalnych: {p.name}')
    return A,unit(N)

def data_for(mode):
    if mode=='synthetic': return synth()
    if mode not in ('full','roi','blind_17','blind_42','blind_81'): raise ValueError('Nieznany tryb')
    d={p.stem:(*read(p),[]) for p in sorted(FULL.glob('brick_part*.ply'))}
    if len(d)!=6: raise ValueError(f'Oczekiwano 6 plików w {FULL}, znaleziono {len(d)}')
    if mode=='roi':
        for p in sorted(ROI.glob('*.ply')):
            parent=p.stem.split('__')[0]
            if parent in d: d[parent][2].append(read(p))
        if sum(bool(x[2]) for x in d.values())<2: raise ValueError('Dodaj ROI co najmniej dwóch części: patches_vienna/brick_part01__roi01.ply itd.')
    else:
        for k,(p,n,r) in d.items(): r.append((p,n))
    if mode.startswith('blind_'):
        rng=np.random.default_rng(int(mode.split('_')[1])); reference={}; out={}
        spacing=max(np.linalg.norm(np.ptp(p,axis=0)) for p,n,r in d.values())
        for k,(p,n,rs) in d.items():
            T=np.eye(4); T[:3,:3]=Rotation.random(random_state=rng).as_matrix()
            T[:3,3]=rng.uniform(-3*spacing,3*spacing,3)-T[:3,:3]@p.mean(0)
            out[k]=(moved(p,T),n@T[:3,:3].T,[(moved(q,T),m@T[:3,:3].T) for q,m in rs]); reference[k]=np.linalg.inv(T).tolist()
        return out,reference
    return d,None

def cloud(p,n):
    c=o3d.geometry.PointCloud(); c.points=o3d.utility.Vector3dVector(p); c.normals=o3d.utility.Vector3dVector(n); return c

def down(p,n,v):
    c=cloud(p,n).voxel_down_sample(v); return np.asarray(c.points).copy(),unit(np.asarray(c.normals).copy())

def features(p,n,v):
    c=cloud(p,n); f=REG.compute_fpfh_feature(c,o3d.geometry.KDTreeSearchParamHybrid(radius=5*v,max_nn=100)); return c,f

def prepare(d,v):
    out={}
    for k,(p,n,rs) in d.items():
        A,N=down(p,n,v); out[k]=(A,N,[(i,*down(q,m,v)) for i,(q,m) in enumerate(rs) if len(q)>=30])
    return out

def pair(a,b,data,v,mode):
    A,NA,ra=data[a]; B,NB,rb=data[b]; eps=1.5*v; candidates=[]; descriptors={}
    for ia,P,NP in ra:
        for ib,Q,NQ in rb:
            if min(len(P),len(Q))<30: continue
            pa,fa=features(P,NP,v); pb,fb=features(Q,NQ,v); _,flip=features(Q,-NQ,v)
            DA=describe(P,NP,v); DB=describe(Q,NQ,v); descriptors[(ia+1,ib+1)]=(DA,DB)
            starts,diag=signature_candidates(DA,DB,v)
            local,ld=local_proposals(P,NP,Q,NQ,v); diag.update(ld); starts=local+starts
            local_count=len(local); signature_count=len(starts); signature_tree=cKDTree(DA[0])
            if mode=='full': starts.append(np.eye(4))
            # PCA orientation hypotheses for known ROI; no segmentation or known truth.
            if mode!='full':
                _,UA=np.linalg.eigh(np.cov(P.T)); _,UB=np.linalg.eigh(np.cov(Q.T))
                for signs in itertools.product((-1,1),repeat=3):
                    R=UA@np.diag(signs)@UB.T
                    if np.linalg.det(R)>0:
                        T=np.eye(4); T[:3,:3]=R; T[:3,3]=P.mean(0)-R@Q.mean(0); starts.append(T)
            for f in (flip,fb):
                for repeat in range(4):
                    r=REG.registration_ransac_based_on_feature_matching(pb,pa,f,fa,True,eps,REG.TransformationEstimationPointToPoint(False),3,[REG.CorrespondenceCheckerBasedOnEdgeLength(.85),REG.CorrespondenceCheckerBasedOnDistance(eps)],REG.RANSACConvergenceCriteria(12000,.995))
                    if r.fitness>0: starts.append(r.transformation)
            for start_index,T0 in enumerate(starts):
                # Retain both raw hypotheses and local refinement; ICP cannot enforce complementary contact.
                ref=REG.registration_icp(pb,pa,eps,T0,REG.TransformationEstimationPointToPlane(),REG.ICPConvergenceCriteria(max_iteration=35))
                for T in (T0,ref.transformation,refine_contact(P,NP,Q,NQ,T0,eps),refine_contact(P,NP,Q,NQ,ref.transformation,eps)):
                    m=metrics(P,NP,Q,NQ,T,eps); pen=penetration(A,NA,B,NB,T,eps)
                    # Signature consistency evaluated only on actual opposing contact.
                    dd,ii=signature_tree.query(moved(DB[0],T)); mask=(dd<2*v)&(np.sum(DA[1][ii]*(DB[1]@T[:3,:3].T),axis=1)<-.65)
                    bf=DB[2][mask].copy(); bf[:,0::5]*=-1; bf[:,3::5]*=-1
                    sig=float(np.median(np.linalg.norm(DA[2][ii[mask]]-bf,axis=1))) if mask.any() else None
                    score=m['quality']/(1+(.5*sig if sig is not None else 0))-3*pen
                    eligible=(m['opposed']>=.65 and min(m['opposed_a'],m['opposed_b'])>= (.025 if mode=='full' or mode.startswith('blind_') else .15) and m['contact_points']>=25 and pen<=.04 and score>0)
                    candidates.append(dict(a=a,b=b,roi_a=ia+1,roi_b=ib+1,T=T.tolist(),score=float(score),penetration=pen,eligible=bool(eligible),signature_error=sig,origin=("local_patch" if start_index<local_count else "signature" if start_index<signature_count else "baseline"),**diag,**m))
    candidates.sort(key=lambda x:(x['eligible'],x['score']),reverse=True)
    kept=[]
    for e in candidates:
        T=np.asarray(e['T'])
        if any(q['roi_a']==e['roi_a'] and q['roi_b']==e['roi_b'] and Rotation.from_matrix(T[:3,:3]@np.asarray(q['T'])[:3,:3].T).magnitude()<.15 and np.linalg.norm(moved(B.mean(0)[None],T)-moved(B.mean(0)[None],np.asarray(q['T'])))<2*v for q in kept): continue
        e['hypothesis_id']=f'{a}|{b}|{len(kept)+1}'
        DA,DB=descriptors[(e['roi_a'],e['roi_b'])]
        d,j=cKDTree(DA[0]).query(moved(DB[0],T)); m=(d<1.5*v)&(np.sum(DA[1][j]*(DB[1]@T[:3,:3].T),axis=1)<-.866)
        ids=np.flatnonzero(m); ids=ids[np.linspace(0,len(ids)-1,min(180,len(ids)),dtype=int)] if len(ids) else ids
        # Middle scale (5v), undo feature weights for displayed proxy.
        e['signature_plot']={'h_a':(DA[2][j[ids],8]*2/v).tolist(),'h_b':(-DB[2][ids,8]*2/v).tolist(),'k_a':(DA[2][j[ids],9]*4/v**2).tolist(),'k_b':(DB[2][ids,9]*4/v**2).tolist(),'distance':d[ids].tolist()}
        kept.append(e)
        if len(kept)>=12: break
    return kept

def poses_for(d,poses,components):
    # Separate unrelated components in display only; not recovered relative positions.
    gap=max(np.linalg.norm(np.ptp(p,axis=0)) for p,n,r in d.values())*.35
    cursor=0.; output={}
    for group in components:
        allp=np.vstack([moved(d[k][0],poses[k]) for k in group]); lo=allp.min(0); hi=allp.max(0); shift=np.array([cursor-lo[0],-(lo[1]+hi[1])/2,-(lo[2]+hi[2])/2]); cursor+=hi[0]-lo[0]+gap
        for k in group:
            T=poses[k].copy(); T[:3,3]+=shift; output[k]=T
    center=cursor/2
    for T in output.values(): T[0,3]-=center
    return output

def initial(d,mode=None):
    if mode and mode.startswith("blind_"): return {k:np.eye(4) for k in d}
    rng=np.random.default_rng(81); spacing=max(np.linalg.norm(np.ptp(p,axis=0)) for p,n,r in d.values())*1.25; out={}
    for i,(k,(p,n,r)) in enumerate(d.items()):
        T=np.eye(4); T[:3,:3]=Rotation.random(random_state=rng).as_matrix(); T[:3,3]=np.array([(i%3-1)*spacing,(i//3-.5)*spacing,0])-T[:3,:3]@p.mean(0); out[k]=T
    return out

def frames(d,start,end):
    out=[]
    for t in np.linspace(0,len(d),len(d)*12+1):
        f={}
        for i,k in enumerate(d):
            u=float(np.clip(t-i,0,1)); a,b=start[k],end[k]; T=np.eye(4); T[:3,:3]=Slerp([0,1],Rotation.from_matrix(np.stack([a[:3,:3],b[:3,:3]])))([u]).as_matrix()[0]; c=d[k][0].mean(0); T[:3,3]=(1-u)*moved(c[None],a)[0]+u*moved(c[None],b)[0]-T[:3,:3]@c; f[k]=T.tolist()
        out.append(f)
    return out

def worker(job,mode):
    try:
        d,truth=data_for(mode); v=min(2*np.max(np.linalg.norm(p-p.mean(0),axis=1)) for p,n,r in d.values())/45; work=prepare(d,v); edges=[]; pairs=list(itertools.combinations(d,2))
        for i,(a,b) in enumerate(pairs):
            job.update(done=i,total=len(pairs),message=f'{a} ↔ {b}'); e=pair(a,b,work,v,mode)
            if e: edges.extend(e)
        job.update(done=len(pairs),message='Beam Search i wspólna optymalizacja')
        poses,components,chosen=beam_assemble(work,edges,1.5*v,unique_roi=mode in ('roi','synthetic'),width=20)
        poses,history=optimize_poses(work,poses,components,chosen,1.5*v)
        end=poses_for(d,poses,components); start=initial(d,mode)
        result=dict(mode=mode,optimization=history,chosen_ids=[e['hypothesis_id'] for e in chosen],voxel=v,eps=1.5*v,edges=sorted(edges,key=lambda x:-x['score']),chosen=[[e['a'],e['b']] for e in chosen],components=components,poses={k:t.tolist() for k,t in poses.items()},frames=frames(d,start,end),warning='Hipotezy eksperymentalne. Normalne muszą być zewnętrzne; wskaźnik przenikania nie jest objętością. Położenie między grupami jest nieznane.')
        # Measure final geometry separately from candidate-local scores.
        result['final_checks']=[]
        for group in components:
            for a,b in itertools.combinations(group,2):
                A,NA,_=work[a]; B,NB,_=work[b]; T=np.linalg.inv(poses[a])@poses[b]
                result['final_checks'].append(dict(a=a,b=b,penetration=penetration(A,NA,B,NB,T,1.5*v),**metrics(A,NA,B,NB,T,1.5*v)))
        if truth and mode=='synthetic':
            truepairs={frozenset((f'synthetic_{i:02d}',f'synthetic_{i+1:02d}')) for i in range(1,6)}; selected={frozenset(x) for x in result['chosen']}; result['validation']={'correct':len(selected&truepairs),'selected':len(selected),'expected':5}
            if len(components)==1:
                anchor=sorted(d)[0]; align=np.asarray(truth[anchor])@np.linalg.inv(poses[anchor]); errs=[]
                for k,(p,n,r) in d.items():
                    if k==anchor: continue
                    T=align@poses[k]; G=np.asarray(truth[k]); X=moved(p,T); Y=moved(p,G)
                    errs.append([float(np.degrees(Rotation.from_matrix(T[:3,:3]@G[:3,:3].T).magnitude())),float(np.sqrt(np.mean(np.sum((X-Y)**2,axis=1))))])
                result['validation']['mean_rotation_deg'],result['validation']['mean_pose_rmse']=np.mean(errs,axis=0).tolist()
        if mode.startswith('blind_'):
            result['blind_test']={'seed':int(mode.split('_')[1]),'input_scrambled':True,'identity_initialization':False,'reference_kind':'Original approximate Vienna poses, not verified ground truth','connected':max(map(len,components)),'components':len(components)}
            if len(components)==1:
                anchor=sorted(d)[0]; align=np.asarray(truth[anchor])@np.linalg.inv(poses[anchor]); errs=[]
                for k,(p,n,r) in d.items():
                    if k==anchor: continue
                    T=align@poses[k]; G=np.asarray(truth[k]); X=moved(p,T); Y=moved(p,G)
                    errs.append(dict(fragment=k,rotation_deg=float(np.degrees(Rotation.from_matrix(T[:3,:3]@G[:3,:3].T).magnitude())),position_rmse=float(np.sqrt(np.mean(np.sum((X-Y)**2,axis=1))))))
                result['blind_test']['deviation_from_approximate_reference']=errs
                result['blind_test']['mean_rotation_deg']=float(np.mean([e['rotation_deg'] for e in errs])); result['blind_test']['mean_position_rmse']=float(np.mean([e['position_rmse'] for e in errs]))
        path=ROOT/'wyniki_osteoform_v6'; path.mkdir(exist_ok=True); (path/(job['id']+'.json')).write_text(json.dumps({k:v for k,v in result.items() if k!='frames'},ensure_ascii=False,indent=2),encoding='utf8'); job.update(state='done',done=len(pairs),result=result,message='Zakończono')
    except Exception as ex: traceback.print_exc(); job.update(state='error',message=str(ex))

@app.get('/')
def index(): return render_template('osteoform_v6.html')
@app.get('/plotly.js')
def plotly_js(): return Response(get_plotlyjs(),mimetype='application/javascript')
@app.get('/api/data')
def get_data():
    try:
        mode=request.args.get('mode','full'); d,_=data_for(mode); start=initial(d,mode); result=[]
        for k,(p,n,r) in d.items():
            ix=np.linspace(0,len(p)-1,min(7500,len(p)),dtype=int); result.append(dict(name=k,points=p[ix].round(5).tolist(),count=len(p),rois=len(r),start=start[k].tolist()))
        return jsonify(result)
    except Exception as e: return jsonify(error=str(e)),400
@app.post('/api/run')
def run():
    mode=request.get_json()['mode']
    with lock:
        if any(j['state']=='running' for j in jobs.values()): return jsonify(error='Analiza już trwa'),409
        key=uuid.uuid4().hex; job=dict(id=key,state='running',done=0,total=15,message='Przygotowanie danych'); jobs[key]=job
    threading.Thread(target=worker,args=(job,mode),daemon=True).start(); return jsonify(id=key)
@app.get('/api/job/<key>')
def status(key): return jsonify(jobs.get(key,{'state':'error','message':'Nieznane zadanie'}))
if __name__=='__main__': app.run(host='127.0.0.1',port=5005,debug=False,threaded=True)
