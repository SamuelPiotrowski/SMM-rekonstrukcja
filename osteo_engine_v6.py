"""Experimental rigid assembly. No fracture segmentation; supplied ROI or full scans.
All distances in input units. Signed penetration is a local proxy, not volume.
"""
import itertools
import numpy as np
from scipy.spatial import cKDTree


def moved(p,T): return p@T[:3,:3].T+T[:3,3]
def unit(n): return n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)

def metrics(A,NA,B,NB,T,eps):
    B=moved(B,T); NB=NB@T[:3,:3].T
    da,ia=cKDTree(B).query(A); db,ib=cKDTree(A).query(B)
    ca=da<=eps; cb=db<=eps
    va=(np.linalg.norm(NA,axis=1)>.5)&(np.linalg.norm(NB[ia],axis=1)>.5)
    vb=(np.linalg.norm(NB,axis=1)>.5)&(np.linalg.norm(NA[ib],axis=1)>.5)
    oa=ca&va&(np.sum(NA*NB[ia],axis=1)<-.866)
    ob=cb&vb&(np.sum(NB*NA[ib],axis=1)<-.866)
    d=np.r_[da[oa],db[ob]]
    # Opposing contact; bidirectional fractions are point-based, not physical area.
    c1,c2=float(oa.mean()),float(ob.mean())
    rmse=float(np.sqrt(np.mean(d*d))) if len(d) else None
    return dict(coverage_a=float(ca.mean()),coverage_b=float(cb.mean()),opposed_a=c1,opposed_b=c2,
                opposed=float((oa.sum()+ob.sum())/max(1,ca.sum()+cb.sum())),rmse=rmse,
                contact_points=int(min(oa.sum(),ob.sum())),quality=float(np.sqrt(c1*c2)/(1+(rmse/eps if rmse is not None else 10))))

def penetration(A,NA,B,NB,T,eps):
    # Nearest tangent plane signed distance; use only near surface samples.
    # Requires consistently OUTWARD normals; concave edges can yield false positives.
    B=moved(B,T); NB=NB@T[:3,:3].T
    def one(P,Q,N):
        d,i=cKDTree(Q).query(P); signed=np.sum((P-Q[i])*N[i],axis=1)
        return float(np.mean((d<4*eps)&(signed< -eps)&(np.linalg.norm(N[i],axis=1)>.5)))
    return max(one(A,B,NB),one(B,A,NA))

def rigid(A,B):
    a,b=A.mean(0),B.mean(0); U,_,Vt=np.linalg.svd((B-b).T@(A-a)); D=np.eye(3); D[-1,-1]=np.linalg.det(Vt.T@U.T)
    R=Vt.T@D@U.T; T=np.eye(4); T[:3,:3]=R; T[:3,3]=a-R@b; return T

def synth():
    # Six adjacent blocks separated by shared nonlinear surfaces; deterministic reference.
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(23); y,z=np.meshgrid(np.linspace(-2,2,35),np.linspace(-2,2,35)); y=y.ravel();z=z.ravel()
    def boundary(i):
        x=3*i+(.22*np.sin((i+1)*y+.8*z)+.16*np.cos(1.7*z-i*y) if 0<i<6 else np.zeros_like(y))
        p=np.c_[x,y,z]; dxdy=.22*(i+1)*np.cos((i+1)*y+.8*z)+.16*i*np.sin(1.7*z-i*y) if 0<i<6 else np.zeros_like(y); dxdz=.176*np.cos((i+1)*y+.8*z)-.272*np.sin(1.7*z-i*y) if 0<i<6 else np.zeros_like(y)
        return p,unit(np.c_[np.ones_like(y),-dxdy,-dxdz])
    out={}; truth={}
    for i in range(6):
        L,nl=boundary(i); R,nr=boundary(i+1); p=[L,R]; n=[-nl,nr]
        for axis in (1,2):
            for sign in (-1,1):
                q=np.linspace(-2,2,35); other=3-axis
                yy=np.full(35,2*sign) if axis==1 else q; zz=q if axis==1 else np.full(35,2*sign)
                def xx(j): return 3*j+(.22*np.sin((j+1)*yy+.8*zz)+.16*np.cos(1.7*zz-j*yy) if 0<j<6 else np.zeros_like(q))
                u=np.linspace(0,1,25)[:,None]; X=(1-u)*xx(i)+u*xx(i+1); P=np.zeros((25,35,3)); P[:,:,0]=X; P[:,:,1]=yy; P[:,:,2]=zz; N=np.zeros_like(P); N[:,:,axis]=sign; p.append(P.reshape(-1,3));n.append(N.reshape(-1,3))
        T=np.eye(4); T[:3,:3]=Rotation.random(random_state=rng).as_matrix(); T[:3,3]=rng.uniform(-5,5,3)
        name=f'synthetic_{i+1:02d}'; truth[name]=np.linalg.inv(T).tolist(); out[name]=(moved(np.vstack(p),T),np.vstack(n)@T[:3,:3].T,[(moved(L,T),-nl@T[:3,:3].T),(moved(R,T),nr@T[:3,:3].T)])
        # Known ROI are supplied by the generator, not detected.
        out[name]=(out[name][0],out[name][1], [r for j,r in enumerate(out[name][2]) if (j==0 and i>0) or (j==1 and i<5)])
    return out,truth

def assemble(data, edges, eps, unique_roi=False):
    poses={k:np.eye(4) for k in data}; groups={k:{k} for k in data}; chosen=[]; used=set()
    for e in sorted(edges,key=lambda x:-x['score']):
        if not e['eligible']: continue
        a,b=e['a'],e['b']; ga,gb=groups[a],groups[b]
        keys={(a,e.get('roi_a')),(b,e.get('roi_b'))}
        if unique_roi and keys & used: e['global_reject']='ROI już wykorzystane'; continue
        if ga is gb: continue
        delta=poses[a]@np.array(e['T'])@np.linalg.inv(poses[b]); worst=0
        for i,j in itertools.product(ga,gb):
            A,NA,_=data[i]; B,NB,_=data[j]; rel=np.linalg.inv(poses[i])@delta@poses[j]
            worst=max(worst,penetration(A,NA,B,NB,rel,eps))
        if worst>.08: e['global_reject']='Przenikanie lokalne w grupie'; continue
        for j in gb: poses[j]=delta@poses[j]
        g=ga|gb
        for j in g: groups[j]=g
        chosen.append(e); used.update(keys)
    components=sorted({tuple(sorted(g)) for g in groups.values()})
    return poses,components,chosen


def refine_contact(P,NP,Q,NQ,T,eps,steps=12):
    """Trimmed, opposing-normal ICP; never reflects or scales a fragment."""
    tree=cKDTree(P); T=T.copy()
    for _ in range(steps):
        X=moved(Q,T); N=NQ@T[:3,:3].T; d,i=tree.query(X)
        mask=(d<2*eps)&(np.sum(N*NP[i],axis=1)<-.65)
        j=np.flatnonzero(mask)
        if len(j)<12: break
        j=j[np.argsort(d[j])[:max(12,int(.85*len(j)))]]
        D=rigid(P[i[j]],X[j]); T=D@T
        if np.linalg.norm(D-np.eye(4))<1e-6: break
    return T


def beam_assemble(data, edges, eps, unique_roi=False, width=10, group_cost=None, accept=None):
    """Bounded beam over alternative rigid group merges; no forced connectivity."""
    names=sorted(data); edges=[e for e in edges if e['eligible']]
    # Fixed sample budget for cross-group local collision checks.
    sample={k:(p[np.linspace(0,len(p)-1,min(1800,len(p)),dtype=int)],n[np.linspace(0,len(p)-1,min(1800,len(p)),dtype=int)]) for k,(p,n,_) in data.items()}
    beam=[(0.,{k:np.eye(4) for k in names},tuple((k,) for k in names),[],frozenset())]; best=beam[0]
    for depth in range(len(names)-1):
        pool=[]
        for score,poses,groups,chosen,used in beam:
            lookup={k:g for g in groups for k in g}
            for e in edges:
                a,b=e['a'],e['b']; ga,gb=lookup[a],lookup[b]
                keys=frozenset(((a,e['roi_a']),(b,e['roi_b'])))
                if ga==gb or (unique_roi and used&keys): continue
                delta=poses[a]@np.asarray(e['T'])@np.linalg.inv(poses[b]); worst=0.
                for i,j in itertools.product(ga,gb):
                    A,NA=sample[i]; B,NB=sample[j]
                    worst=max(worst,penetration(A,NA,B,NB,np.linalg.inv(poses[i])@delta@poses[j],eps))
                    if worst>.06: break
                if worst>.06: continue
                nxt=dict(poses)
                for j in gb: nxt[j]=delta@poses[j]
                gs=tuple(sorted([g for g in groups if g not in (ga,gb)]+[tuple(sorted(ga+gb))]))
                change = 0. if group_cost is None else group_cost(ga+gb,nxt) - group_cost(ga,poses) - group_cost(gb,poses)
                pool.append((score+e['score']-2*worst-change,nxt,gs,chosen+[e],used|keys))
        if not pool: break
        pool.sort(key=lambda s:s[0],reverse=True); seen=set(); beam=[]
        for s in pool:
            sig=tuple(sorted(e['hypothesis_id'] for e in s[3]))
            if sig in seen: continue
            seen.add(sig); beam.append(s)
            if len(beam)>=width: break
        for candidate in beam:
            if candidate[0] <= best[0]: break
            if accept is None or accept(candidate[1], candidate[2]):
                best=candidate
                break
    return best[1],best[2],best[3]


def optimize_poses(data,poses,components,chosen,eps):
    """Joint rigid contact refinement with robust loss and all-pair guard.
    Fixed nearest opposing correspondences per outer iteration; no truth used.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    poses={k:t.copy() for k,t in poses.items()}; history=[]
    for group in components:
        if len(group)<2: continue
        anchor=group[0]; free=list(group[1:]); links=[e for e in chosen if e['a'] in group and e['b'] in group]
        samples={k:(data[k][0][::max(1,len(data[k][0])//2000)],data[k][1][::max(1,len(data[k][0])//2000)]) for k in group}
        def collision(ps):
            return max((penetration(*samples[a],*samples[b],np.linalg.inv(ps[a])@ps[b],eps) for a,b in itertools.combinations(group,2)),default=0.)
        for iteration in range(4):
            corr=[]
            for e in links:
                a,b=e['a'],e['b']; P,NP=next((p,n) for i,p,n in data[a][2] if i+1==e['roi_a']); Q,NQ=next((p,n) for i,p,n in data[b][2] if i+1==e['roi_b']); X=moved(P,poses[a]); Y=moved(Q,poses[b]); d,j=cKDTree(X).query(Y); n1=NP@poses[a][:3,:3].T; n2=NQ@poses[b][:3,:3].T
                ids=np.flatnonzero((d<eps*1.5)&(np.sum(n1[j]*n2,axis=1)<-.7)); ids=ids[np.argsort(d[ids])[:min(500,max(0,int(.9*len(ids))))]]
                if len(ids)>=12: corr.append((a,b,P[j[ids]],Q[ids],NP[j[ids]],NQ[ids]))
            if not corr: break
            def unpack(x):
                out=dict(poses)
                for i,k in enumerate(free):
                    D=np.eye(4); D[:3,:3]=Rotation.from_rotvec(x[6*i:6*i+3]).as_matrix(); D[:3,3]=x[6*i+3:6*i+6]*eps; out[k]=D@poses[k]
                return out
            def residual(x):
                ps=unpack(x); r=[]
                for a,b,P,Q,N,M in corr:
                    r.extend(((moved(P,ps[a])-moved(Q,ps[b]))/eps).ravel()); r.extend((.3*(N@ps[a][:3,:3].T+M@ps[b][:3,:3].T)).ravel())
                return np.r_[r,.05*x]
            x=np.zeros(6*len(free)); before=float(np.sqrt(np.mean(residual(x)**2))); fit=least_squares(residual,x,loss='soft_l1',f_scale=.5,max_nfev=15); after=float(np.sqrt(np.mean(residual(fit.x)**2))); trial=unpack(fit.x); old=collision(poses); new=collision(trial); accept=after<before and new<=max(old+.005,.04) and new<=.08
            history.append(dict(group=anchor,iteration=iteration+1,before=before,after=after,collision_before=old,collision_after=new,accepted=bool(accept)))
            if not accept: break
            poses=trial
    return poses,history
