
import argparse, csv, os
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import zoom, gaussian_filter, affine_transform
from scipy.optimize import linear_sum_assignment

CFG=dict(size=64,trim=0.75,ssc_sigma=1.0); RNG=np.random.default_rng(0)
_AX=[(1,0,0),(-1,0,0)];_AY=[(0,1,0),(0,-1,0)];_AZ=[(0,0,1),(0,0,-1)]
SSC12=[(p,q) for A,B in [(_AX,_AY),(_AX,_AZ),(_AY,_AZ)] for p in A for q in B]
def fix_path(root,rel):
    p=Path(rel) if os.path.isabs(rel) else root/rel
    if p.exists():return p
    if str(p).endswith(".nii.gz"):
        a=Path(str(p)[:-3]);return a if a.exists() else p
    return p
def read(p):return list(csv.DictReader(open(p,newline="")))
def load_raw(path,size):
    v=np.squeeze(np.nan_to_num(nib.load(str(path)).get_fdata().astype(np.float32)))
    v=zoom(v,[size/s for s in v.shape],order=1)
    pos=v[v>0];lo,hi=(np.percentile(pos,[1,99]) if pos.size else (0,1))
    return np.clip((v-lo)/(hi-lo+1e-8),0,1).astype(np.float32)
def rand_rigid(v):                        # BREAK co-location: independent rigid per volume, NO re-registration after
    ang=RNG.uniform(-0.35,0.35,3);c=np.cos(ang);s=np.sin(ang)
    Rx=np.array([[1,0,0],[0,c[0],-s[0]],[0,s[0],c[0]]]);Ry=np.array([[c[1],0,s[1]],[0,1,0],[-s[1],0,c[1]]])
    Rz=np.array([[c[2],-s[2],0],[s[2],c[2],0],[0,0,1]]);M=Rz@Ry@Rx
    off=np.array(v.shape)/2-M@(np.array(v.shape)/2)+RNG.uniform(-8,8,3)
    return affine_transform(v,M,offset=off,order=1).astype(np.float32)
def ssc_field(vol,s=1.0):
    v=gaussian_filter(vol,s)
    d=np.stack([gaussian_filter((np.roll(v,p,(0,1,2))-np.roll(v,q,(0,1,2)))**2,s) for p,q in SSC12])
    de=np.exp(-d/(d.mean(0,keepdims=True)+1e-6));return (de/(np.linalg.norm(de,axis=0,keepdims=True)+1e-8)).astype(np.float32)
def trimmed_S(qids,gids,qv,gv,qd,gd,T):
    n,m=len(qids),len(gids);S=np.zeros((n,m),np.float32)
    for i,qi in enumerate(qids):
        for j,gj in enumerate(gids):
            msk=(qv[qi]>0.05)|(gv[gj]>0.05);dv=np.sort(np.abs(qd[qi]-gd[gj]).mean(0)[msk]);k=max(1,int(round((1-T)*dv.size)))
            S[i,j]=-float(dv[:k].mean()) if dv.size else 0.0
    return S
def hung(S,qids,gids):
    r,c=linear_sum_assignment(-S);asg={qids[i]:gids[j] for i,j in zip(r,c)};o={}
    for i,q in enumerate(qids):a=asg[q];o[q]=[a]+[gids[j] for j in np.argsort(-S[i]) if gids[j]!=a]
    return o
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-root",required=True);ap.add_argument("--out",default="submission_d3_colocation_broken.csv");a=ap.parse_args()
    root=Path(a.data_root).resolve();rows=[]
    for sp in ["val","test"]:
        qc=read(root/f"dataset3/{sp}_queries.csv");gc=read(root/f"dataset3/{sp}_gallery.csv")
        qids=[r["query_id"] for r in qc];gids=[r["target_id"] for r in gc]
        qimg={r["query_id"]:fix_path(root,r["query_image"]) for r in qc};gimg={r["target_id"]:fix_path(root,r["target_image"]) for r in gc}
        print(f"dataset3/{sp}: co-location BROKEN (random rigid), NO re-registration ...",flush=True)
        qv={k:load_raw(p,CFG["size"]) for k,p in qimg.items()}
        gv={k:rand_rigid(load_raw(p,CFG["size"])) for k,p in gimg.items()}   # break alignment, do NOT register back
        qd={k:ssc_field(v) for k,v in qv.items()};gd={k:ssc_field(v) for k,v in gv.items()}
        S=trimmed_S(qids,gids,qv,gv,qd,gd,CFG["trim"]);h=hung(S,qids,gids)
        for q in qids:rows.append((q," ".join(h[q])))
    with open(a.out,"w",newline="") as f:
        w=csv.writer(f);w.writerow(["query_id","target_id_ranking"])
        for q,r in rows:w.writerow([q,r])
    print(f"wrote {len(rows)} rows -> {a.out} (co-location broken, no re-reg)",flush=True)
if __name__=="__main__":main()
