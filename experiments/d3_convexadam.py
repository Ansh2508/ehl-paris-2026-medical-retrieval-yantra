
import argparse, csv, os, time
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

CFG = dict(size=96, trim=0.75, grid_sp=4, disp_hw=4, adam_iters=40, lambda_reg=0.7)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, flush=True)

def fix_path(root, rel):
    p = Path(rel) if os.path.isabs(rel) else root/rel
    if p.exists(): return p
    if str(p).endswith(".nii.gz"):
        alt = Path(str(p)[:-3])
        if alt.exists(): return alt
    return p
def read(p): return list(csv.DictReader(open(p, newline="")))

def load_raw(path, size):                       # own array + cube: never uses the query's shared frame
    v = nib.load(str(path)).get_fdata().astype(np.float32); v = np.squeeze(np.nan_to_num(v))
    import scipy.ndimage as ndi
    v = ndi.zoom(v, [size/s for s in v.shape], order=1)
    pos = v[v > 0]; lo, hi = (np.percentile(pos,[1,99]) if pos.size else (0,1))
    return np.clip((v-lo)/(hi-lo+1e-8), 0, 1).astype(np.float32)

# ---- MIND-SSC descriptor (torch, GPU) ----
def mindssc(img, radius=2, dilation=2):
    # img: (1,1,H,W,D)
    def pdist(x, y): return (x-y)**2
    H,W,D = img.shape[-3:]
    six = torch.tensor([[0,1,1],[1,1,0],[1,0,1],[1,1,2],[2,1,1],[1,2,1]],device=img.device)
    kernel = torch.tensor([[[0.,0,0],[0,1,0],[0,0,0]]],device=img.device)
    # 12 SSC edges via shifts
    shift = []
    base = [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]
    d = dilation
    pts = [(d*a,d*b,d*c) for a,b,c in base]
    pairs = [(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)]
    feats=[]
    img_p = img
    for i,j in pairs:
        a = torch.roll(img_p, pts[i], dims=(-3,-2,-1))
        b = torch.roll(img_p, pts[j], dims=(-3,-2,-1))
        feats.append(F.avg_pool3d((a-b)**2, 3, stride=1, padding=1))
    mind = torch.cat(feats,1)                        # (1,12,H,W,D)
    mind = mind - mind.min(1,keepdim=True)[0]
    var = mind.mean(1,keepdim=True).clamp_min(1e-6)
    mind = torch.exp(-mind/var)
    mind = mind / (mind.norm(dim=1,keepdim=True)+1e-8)
    return mind

def correlate(mf, mm, disp_hw, gsp):
    # discretised-displacement SSD cost volume over a (2*disp_hw+1)^3 search
    H,W,D = mf.shape[-3:]
    mm_p = F.pad(mm, (disp_hw,)*6)
    C = mf.shape[1]
    ssd = []
    rng = range(-disp_hw, disp_hw+1)
    for dz in rng:
      for dy in rng:
        for dx in rng:
            shifted = mm_p[..., disp_hw+dz:disp_hw+dz+H, disp_hw+dy:disp_hw+dy+W, disp_hw+dx:disp_hw+dx+D]
            # align spatial dims
            shifted = torch.roll(mm, shifts=(dz,dy,dx), dims=(-3,-2,-1))
            ssd.append(((mf-shifted)**2).mean(1,keepdim=True))
    ssd = torch.cat(ssd,1)                             # (1, K, H,W,D)
    return ssd

def coupled_convex(ssd, disp_hw):
    # soft global regularisation: argmin then smooth the displacement field
    K = ssd.shape[1]; H,W,D = ssd.shape[-3:]
    rng = torch.arange(-disp_hw, disp_hw+1, device=ssd.device, dtype=torch.float32)
    gz,gy,gx = torch.meshgrid(rng,rng,rng, indexing='ij')
    disp_grid = torch.stack([gx.reshape(-1),gy.reshape(-1),gz.reshape(-1)],0)   # (3,K)
    soft = F.softmax(-ssd*150.0, dim=1)                # (1,K,H,W,D)
    disp = torch.einsum('ck,bkhwd->bchwd', disp_grid, soft)   # (1,3,H,W,D) expected displacement
    # smooth
    for _ in range(3):
        disp = F.avg_pool3d(disp, 3, stride=1, padding=1)
    return disp

def adam_refine(mf, mm, disp0, iters, lam):
    disp = disp0.clone().requires_grad_(True)
    opt = torch.optim.Adam([disp], lr=0.5)
    H,W,D = mf.shape[-3:]
    base = F.affine_grid(torch.eye(3,4,device=mf.device).unsqueeze(0), (1,1,H,W,D), align_corners=False)
    for _ in range(iters):
        opt.zero_grad()
        dperm = disp.permute(0,2,3,4,1)
        scale = torch.tensor([ (W-1)/2,(H-1)/2,(D-1)/2 ],device=mf.device)
        grid = base + (dperm.flip(-1)/scale)
        warped = F.grid_sample(mm, grid, align_corners=False, padding_mode='border')
        sim = ((mf-warped)**2).mean()
        reg = sum(((disp[...,1:,:,:]-disp[...,:-1,:,:])**2).mean() for _ in [0]) \
            + ((disp[...,:,1:,:]-disp[...,:,:-1,:])**2).mean() \
            + ((disp[...,:,:,1:]-disp[...,:,:,:-1])**2).mean()
        (sim + lam*reg).backward()
        opt.step()
    return disp.detach()

def warp(vol, disp):
    H,W,D = vol.shape[-3:]
    base = F.affine_grid(torch.eye(3,4,device=vol.device).unsqueeze(0), (1,1,H,W,D), align_corners=False)
    scale = torch.tensor([(W-1)/2,(H-1)/2,(D-1)/2],device=vol.device)
    grid = base + (disp.permute(0,2,3,4,1).flip(-1)/scale)
    return F.grid_sample(vol, grid, align_corners=False, padding_mode='border')

def deformable_to_template(moving_arr, template_t):
    m = torch.from_numpy(moving_arr).to(DEV).view(1,1,*moving_arr.shape)
    mf = mindssc(template_t); mm = mindssc(m)
    mf_lr = F.avg_pool3d(mf, CFG["grid_sp"]); mm_lr = F.avg_pool3d(mm, CFG["grid_sp"])
    ssd = correlate(mf_lr, mm_lr, CFG["disp_hw"], CFG["grid_sp"])
    disp_lr = coupled_convex(ssd, CFG["disp_hw"])
    disp = F.interpolate(disp_lr, size=moving_arr.shape, mode='trilinear', align_corners=False)
    disp = adam_refine(mf, mm, disp, CFG["adam_iters"], CFG["lambda_reg"])
    w = warp(m, disp)
    return w.squeeze().cpu().numpy().astype(np.float32)

# ---- SSC field (numpy, for scoring, same as our champion) ----
import scipy.ndimage as ndi
_AX=[(1,0,0),(-1,0,0)]; _AY=[(0,1,0),(0,-1,0)]; _AZ=[(0,0,1),(0,0,-1)]
SSC12 = [(p,q) for A,B in [(_AX,_AY),(_AX,_AZ),(_AY,_AZ)] for p in A for q in B]
def ssc_field(vol, s=1.0):
    v = ndi.gaussian_filter(vol, s)
    d = np.stack([ndi.gaussian_filter((np.roll(v,p,(0,1,2))-np.roll(v,q,(0,1,2)))**2, s) for p,q in SSC12])
    var = d.mean(0,keepdims=True)+1e-6; de=np.exp(-d/var)
    return (de/(np.linalg.norm(de,axis=0,keepdims=True)+1e-8)).astype(np.float32)

def build_template(arrs):
    return np.mean(np.stack(arrs),axis=0).astype(np.float32)   # neutral mean, deformable handles the rest

def trimmed_S(qids,gids,qv,gv,qd,gd,TRIM):
    n,m=len(qids),len(gids); S=np.zeros((n,m),np.float32)
    for i,qi in enumerate(qids):
        for j,gj in enumerate(gids):
            msk=(qv[qi]>0.05)|(gv[gj]>0.05)
            dv=np.sort(np.abs(qd[qi]-gd[gj]).mean(0)[msk]); k=max(1,int(round((1-TRIM)*dv.size)))
            S[i,j]=-float(dv[:k].mean()) if dv.size else 0.0
    return S

def hungarian_rank(S,qids,gids):
    row,col=linear_sum_assignment(-S); asg={qids[r]:gids[c] for r,c in zip(row,col)}
    out={}
    for i,qid in enumerate(qids):
        a=asg[qid]; rest=[gids[j] for j in np.argsort(-S[i]) if gids[j]!=a]; out[qid]=[a]+rest
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",required=True)
    ap.add_argument("--out",default="submission_dataset3_convexadam.csv"); a=ap.parse_args()
    root=Path(a.data_root).resolve(); rows=[]
    for split in ["val","test"]:
        qc=read(root/f"dataset3/{split}_queries.csv"); gc=read(root/f"dataset3/{split}_gallery.csv")
        qids=[r["query_id"] for r in qc]; gids=[r["target_id"] for r in gc]
        qimg={r["query_id"]:fix_path(root,r["query_image"]) for r in qc}
        gimg={r["target_id"]:fix_path(root,r["target_image"]) for r in gc}
        print(f"dataset3/{split}: deformable (MIND-SSC + convex + Adam) to neutral template ...", flush=True)
        garrs=[load_raw(p,CFG["size"]) for p in gimg.values()]
        tmpl = torch.from_numpy(build_template(garrs)).to(DEV).view(1,1,CFG["size"],CFG["size"],CFG["size"])
        t0=time.time()
        gv={}; 
        for i,(k,p) in enumerate(gimg.items()):
            gv[k]=deformable_to_template(load_raw(p,CFG["size"]), tmpl)
            if i%10==0: print(f"  gal {i}/{len(gimg)} ({time.time()-t0:.0f}s)", flush=True)
        qv={}
        for i,(k,p) in enumerate(qimg.items()):
            qv[k]=deformable_to_template(load_raw(p,CFG["size"]), tmpl)
            if i%10==0: print(f"  qry {i}/{len(qimg)} ({time.time()-t0:.0f}s)", flush=True)
        gd={k:ssc_field(v) for k,v in gv.items()}; qd={k:ssc_field(v) for k,v in qv.items()}
        S=trimmed_S(qids,gids,qv,gv,qd,gd,CFG["trim"]); h=hungarian_rank(S,qids,gids)
        for qid in qids: rows.append((qid," ".join(h[qid])))
        print(f"  dataset3/{split} done ({time.time()-t0:.0f}s)", flush=True)
    with open(a.out,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["query_id","target_id_ranking"])
        for qid,r in rows: w.writerow([qid,r])
    print(f"\nwrote {len(rows)} rows -> {a.out}  (deformable convex+Adam d3, leak-free)", flush=True)

if __name__ == "__main__": main()
