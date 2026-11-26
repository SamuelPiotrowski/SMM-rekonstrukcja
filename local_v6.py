"""Local oriented patch matching, no fracture segmentation or reference poses."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from osteo_engine_v6 import moved, refine_contact


def anchors(P,N,v,limit=440):
    # Deterministic farthest-point sampling, independent of point ordering after first anchor.
    ids=[int(np.argmax(np.linalg.norm(P-P.mean(0),axis=1)))]; ds=np.full(len(P),np.inf)
    for _ in range(min(limit,len(P))-1):
        ds=np.minimum(ds,np.sum((P-P[ids[-1]])**2,axis=1)); j=int(ds.argmax())
        if ds[j]<v*v: break
        ids.append(j)
    X=P[ids]; M=N[ids]; tree=cKDTree(P)
    d,j=tree.query(X,k=min(100,len(P))); V=P[j]-X[:,None]; h=np.einsum('ijk,ik->ij',V,M)/v
    dot=np.einsum('ijk,ik->ij',N[j],M)
    feats=[]; signs=[]
    for lo,hi in [(0,3),(3,6),(6,10)]:
        mask=(d>lo*v)&(d<=hi*v); cnt=mask.sum(1); den=np.maximum(cnt,1)
        avg=(h*mask).sum(1)/den/hi; rms=np.sqrt((h*h*mask).sum(1)/den)/hi
        turn=((1-dot)*mask).sum(1)/den
        signs.append(len(feats)); feats.extend([avg*3,rms*2,turn])
    F=np.column_stack(feats)
    axes=np.eye(3)[np.argmin(abs(M),axis=1)]; U=np.cross(M,axes); U/=np.maximum(np.linalg.norm(U,axis=1)[:,None],1e-12)
    W=np.cross(M,U); frames=np.stack([U,W,M],axis=-1)
    return X,M,F,signs,frames


def proposals(P,NP,Q,NQ,v,keep=32):
    A,NA,F,signs,FA=anchors(P,NP,v); B,NB,G,_,FB=anchors(Q,NQ,v)
    G=G.copy(); G[:,signs]*=-1
    k=min(10,len(A)); _,inds=cKDTree(F).query(G,k=k); inds=np.asarray(inds).reshape(len(B),k)
    # Complementary normal frame: rotate a flipped source normal around target normal.
    angles=np.arange(16)*2*np.pi/16
    Z=Rotation.from_rotvec(np.c_[np.zeros((16,2)),angles]).as_matrix()@np.diag([1,-1,-1])
    ia=inds.ravel(); ib=np.repeat(np.arange(len(B)),k)
    tree=cKDTree(P); coarse=[]; total=0
    # Local verification of the source neighborhood before testing full contact.
    _,nn=cKDTree(Q).query(B,k=min(40,len(Q))); local=Q[nn]-B[:,None]; ln=NQ[nn]
    for start in range(0,len(ia),64):
        i=ia[start:start+64]; j=ib[start:start+64]
        R=FA[i,None]@Z[None]@FB[j,None].transpose(0,1,3,2)
        X=np.einsum('baij,bnj->bani',R,local[j])+A[i,None,None,:]
        NX=np.einsum('baij,bnj->bani',R,ln[j]); d,idx=tree.query(X)
        opp=np.sum(NP[idx]*NX,axis=-1)<-.65
        scores=(np.exp(-(d/v)**2)*opp).mean(-1)
        # Keep more than one orientation per patch pair.
        for row in range(len(i)):
            for ang in np.argsort(scores[row])[-2:]:
                s=float(scores[row,ang]); total+=1
                if s<.12: continue
                r=R[row,ang]; T=np.eye(4); T[:3,:3]=r; T[:3,3]=A[i[row]]-r@B[j[row]]
                coarse.append((s,T))
    # Test spatially diverse local candidates against the entire source fragment.
    coarse.sort(key=lambda x:-x[0]); center=Q.mean(0); diverse=[]; centers=[]
    for s,T in coarse:
        c=moved(center[None],T)[0]
        if any(np.linalg.norm(c-d)<2*v and np.linalg.norm(T[:3,:3]-U[:3,:3])<.28 for d,U in zip(centers,diverse)): continue
        diverse.append(T); centers.append(c)
        if len(diverse)>=320: break
    ids=np.linspace(0,len(Q)-1,min(1000,len(Q)),dtype=int); out=[]
    for T in diverse:
        T=refine_contact(P,NP,Q,NQ,T,2*v,steps=8)
        X=moved(Q[ids],T); d,i=tree.query(X); dots=np.sum(NP[i]*(NQ[ids]@T[:3,:3].T),axis=1)
        score=float(np.mean(np.exp(-(d/(1.5*v))**2)*(dots<-.65)))
        out.append((score,T))
    out.sort(key=lambda x:-x[0]); selected=[]
    for s,T in out:
        if any(np.linalg.norm(moved(center[None],T)-moved(center[None],U))<2*v and np.linalg.norm(T[:3,:3]-U[:3,:3])<.25 for U in selected): continue
        selected.append(T)
        if len(selected)>=keep: break
    return selected,{'local_orientations_tested':len(ia)*16,'local_candidates':len(coarse),'local_starts':len(selected)}
