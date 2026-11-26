"""Local complementary signatures and rigid geometric voting. No ROI detection.
Signed height / normal variation descriptors are not exact principal curvatures.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from osteo_engine_v6 import rigid, moved


def describe(P,N,v,limit=1000):
    # All neighborhoods are measured in original physical units.
    ids=np.linspace(0,len(P)-1,min(limit,len(P)),dtype=int)
    X=P[ids]; M=N[ids]; tree=cKDTree(P)
    d,j=tree.query(X,k=min(96,len(P)))
    V=P[j]-X[:,None]; h=np.einsum('ijk,ik->ij',V,M)
    dot=np.einsum('ijk,ik->ij',N[j],M)
    # Local Monge patch: z = ax²/2 + bxy + cy²/2 + dx + ey.
    # Tangent coordinates are computed from the supplied oriented normal.
    axes=np.eye(3)[np.argmin(np.abs(M),axis=1)]
    U=np.cross(M,axes); U/=np.maximum(np.linalg.norm(U,axis=1)[:,None],1e-12)
    W=np.cross(M,U)
    x=np.einsum('ijk,ik->ij',V,U)/v; y=np.einsum('ijk,ik->ij',V,W)/v
    design=np.stack([.5*x*x,x*y,.5*y*y,x,y],axis=-1)
    feats=[]; signed=[]; unsigned=[]
    for r in (2.5*v,5*v,9*v):
        valid=(d>1e-9)&(d<=r); count=valid.sum(1); denom=np.maximum(count,1)
        avg=(h*valid).sum(1)/denom/r
        rms=np.sqrt(((h/r)**2*valid).sum(1)/denom)
        turn=((1-np.clip(dot,-1,1))*valid).sum(1)/denom
        signed.append(avg); unsigned.extend([rms,turn])
        # Signed radial heights approximate bending at different scales.
        # The local fit is an approximate curvature proxy in tangent coordinates.
        weights=valid*np.exp(-2*(d/r)**2)
        lhs=np.einsum('nki,nk,nkj->nij',design,weights,design)+1e-5*np.eye(5)
        rhs=np.einsum('nki,nk,nk->ni',design,weights,h/v)
        coef=np.linalg.solve(lhs,rhs[...,None])[...,0]
        mean=.5*(coef[:,0]+coef[:,2]); gaussian=coef[:,0]*coef[:,2]-coef[:,1]**2
        mean=np.where(count>=8,np.clip(mean,-2,2),0); gaussian=np.where(count>=8,np.clip(gaussian,-2,2),0)
        # Dimensionless: v*H and v²*K; H flips, K does not.
        feats.extend([3*avg,2*rms,turn,.5*mean,.25*gaussian])
    return X,M,np.column_stack(feats),np.column_stack(signed),np.column_stack(unsigned)


def candidates(A,B,v,seed=17,budget=90000,keep=24):
    P,NP,F,_,_=A; Q,NQ,G,_,_=B
    G=G.copy(); G[:,0::5]*=-1; G[:,3::5]*=-1  # reverse outward normal convention
    rng=np.random.default_rng(seed)
    k=min(8,len(P)); fd,idx=cKDTree(F).query(G,k=k); idx=np.asarray(idx).reshape(len(Q),k); fd=np.asarray(fd).reshape(len(Q),k)
    # Sample triangles from distinctive (nonflat) neighborhoods, while preserving diversity.
    strength=np.linalg.norm(G,axis=1); weights=.15+np.minimum(strength,2); weights/=weights.sum()
    # A source point can have multiple plausible destination points.
    n=budget; b0=rng.choice(len(Q),size=n,p=weights); b1=rng.choice(len(Q),size=n,p=weights); b2=rng.choice(len(Q),size=n,p=weights)
    bs=np.stack([b0,b1,b2],1); ranks=rng.integers(0,k,size=(n,3)); aa=idx[bs,ranks]
    Bp=Q[bs]; Ap=P[aa]; sides=lambda x:np.linalg.norm(x[:,[1,2,0]]-x,axis=2)
    sb,sa=sides(Bp),sides(Ap); delta=np.abs(sa-sb)
    area=np.linalg.norm(np.cross(Bp[:,1]-Bp[:,0],Bp[:,2]-Bp[:,0]),axis=1)
    good=(sb.min(1)>3*v)&(area>5*v*v)&(delta.max(1)<1.6*v)&(np.max(fd[bs,ranks],axis=1)<.8)
    inds=np.flatnonzero(good)
    # Rank by edge agreement and signature distance, then verify with surface contact.
    err=delta.sum(1)/v+fd[bs,ranks].sum(1)
    inds=inds[np.argsort(err[inds])[:1800]]
    # Two correspondences plus a normal define a rigid local frame when
    # the chord is not parallel to the normal. This is a second hypothesis source.
    pairs=np.flatnonzero((sb[:,0]>4*v)&(delta[:,0]<.8*v)&(fd[bs,ranks][:,:2].max(1)<.6))
    pairs=pairs[np.argsort((delta[:,0]/v+fd[bs,ranks][:,:2].sum(1))[pairs])[:1400]]
    tree=cKDTree(P); sample=np.linspace(0,len(Q)-1,min(450,len(Q)),dtype=int); out=[]
    def proposals():
        for j in inds: yield j,rigid(Ap[j],Bp[j]),3
        for j in pairs:
            a,b=aa[j,:2],bs[j,:2]
            # Virtual third point is normal-directed, not an observed surface point.
            if abs(np.dot((P[a[1]]-P[a[0]])/max(sa[j,0],1e-9),NP[a[0]]))>.85: continue
            yield j,rigid(np.vstack([P[a],P[a[0]]+3*v*NP[a[0]]]),np.vstack([Q[b],Q[b[0]]-3*v*NQ[b[0]]])),2
    for j,T,nmatch in proposals():
        R=T[:3,:3]
        if np.mean(np.sum(NP[aa[j,:nmatch]]*(NQ[bs[j,:nmatch]]@R.T),axis=1))>-.55: continue
        d,i=tree.query(moved(Q[sample],T)); opp=np.sum(NP[i]*(NQ[sample]@R.T),axis=1)<-.65
        contact=(d<2*v)&opp; score=float(np.sum(np.exp(-d[contact]/v))/len(sample))
        if score>0: out.append((score,T))
    out.sort(key=lambda x:-x[0]); selected=[]; center=Q.mean(0)
    for score,T in out:
        if any(Rotation.from_matrix(T[:3,:3]@U[:3,:3].T).magnitude()<.12 and np.linalg.norm(moved(center[None],T)-moved(center[None],U))<2*v for U in selected): continue
        selected.append(T)
        if len(selected)>=keep: break
    return selected,dict(triangles_tested=int(n),triangles_consistent=int(good.sum()),signature_starts=len(selected),frame_proposals=len(pairs))


def signature_error(P,NP,Q,NQ,T,v):
    A=describe(P,NP,v,700); B=describe(Q,NQ,v,700)
    X,Y=A[0],moved(B[0],T); d,i=cKDTree(X).query(Y)
    mask=(d<2*v)&(np.sum(A[1][i]*(B[1]@T[:3,:3].T),axis=1)<-.65)
    if not mask.any(): return None
    a=A[2][i[mask]]; b=B[2][mask].copy(); b[:,0::5]*=-1; b[:,3::5]*=-1
    return float(np.median(np.linalg.norm(a-b,axis=1)))
