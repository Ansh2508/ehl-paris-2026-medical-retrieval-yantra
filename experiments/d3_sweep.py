
import argparse, csv, os, time
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom, gaussian_filter, binary_erosion
from scipy.optimize import linear_sum_assignment

CFG = dict(size=64, reg_iters=120, template_iters=80, ssc_sigma=1.0,
           trims=[0.60,0.70,0.75,0.80,0.85])

_AX=[(1,0,0),(-1,0,0)]; _AY=[(0,1,0),(0,-1,0)]; _AZ=[(0,0,1),(0,0,-1)]
SSC12 = [(p,q) for A,B in [(_AX,_AY),(_AX,_AZ),(_AY,_AZ)] for p in A for q in B]
MIND6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

def fix_path(root, rel):
    p = Path(rel) if os.path.isabs(rel) else root/rel
    if p.exists(): return p
    if str(p).endswith(".nii.gz"):
        alt = Path(str(p)[:-3])
        if alt.exists(): return alt
    return p
def read(p): return list(csv.DictReader(open(p, newline="")))

def load_raw(path, size):                       # OWN array + cube resize: never uses the query's shared frame
    v = nib.load(str(path)).get_fdata().astype(np.float32); v = np.squeeze(np.nan_to_num(v))
    v = zoom(v, [size/s for s in v.shape], order=1)
    pos = v[v > 0]; lo, hi = (np.percentile(pos, [1, 99]) if pos.size else (0, 1))
    return np.clip((v - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

def ssc_field(vol, s=1.0):
    v = gaussian_filter(vol, s)
    d = np.stack([gaussian_filter((np.roll(v,p,(0,1,2)) - np.roll(v,q,(0,1,2)))**2, s) for p,q in SSC12])
    var = d.mean(0, keepdims=True) + 1e-6
    de = np.exp(-d/var)
    return (de / (np.linalg.norm(de, axis=0, keepdims=True) + 1e-8)).astype(np.float32)

def mind_field(vol, s=1.0):
    v = gaussian_filter(vol, s)
    d = np.stack([gaussian_filter((v - np.roll(v,o,(0,1,2)))**2, s) for o in MIND6])
    var = d.mean(0, keepdims=True) + 1e-6
    de = np.exp(-d/var)
    return (de / (np.linalg.norm(de, axis=0, keepdims=True) + 1e-8)).astype(np.float32)

def _sitk(arr): return sitk.Cast(sitk.GetImageFromArray(arr), sitk.sitkFloat32)

def _affine_to(fixed_img, moving_arr, iters, mask_arr=None):   # CONTENT MI; optional brain mask on the moving image
    m = _sitk(moving_arr)
    rinit = sitk.CenteredTransformInitializer(fixed_img, m, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS)
    def stage(tx, mi, lr, it):
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.25, seed=1)
        if mask_arr is not None:
            R.SetMetricMovingMask(sitk.Cast(_sitk(mask_arr), sitk.sitkUInt8))   # ignore resection cavity
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsRegularStepGradientDescent(lr, 1e-4, it)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([4,2,1]); R.SetSmoothingSigmasPerLevel([2,1,0])
        if mi is not None: R.SetMovingInitialTransform(mi)
        R.SetInitialTransform(tx, inPlace=False)
        try: return R.Execute(fixed_img, m)
        except Exception: return mi if mi is not None else tx
    rg = sitk.CompositeTransform([stage(sitk.Euler3DTransform(rinit), None, 1.0, 60)])
    total = sitk.CompositeTransform([rg, stage(sitk.AffineTransform(3), rg, 0.5, iters)])
    return sitk.GetArrayFromImage(sitk.Resample(m, fixed_img, total, sitk.sitkLinear, 0.0, m.GetPixelID()))

def brain_mask(vol, thr=0.05):
    return (vol > thr).astype(np.float32)

def build_template(arrs, iters):
    seed = np.mean(np.stack(arrs), axis=0); si=_sitk(seed); w=[]; t0=time.time()
    for i,a in enumerate(arrs):
        w.append(_affine_to(si, a, iters))
        if i%20==0: print(f"  template {i}/{len(arrs)} ({time.time()-t0:.0f}s)", flush=True)
    return np.mean(np.stack(w), axis=0)

def reg_all(imgs, ref_img, iters, tag, masked=False):
    vols={}; t0=time.time()
    for i,(k,p) in enumerate(imgs.items()):
        arr = load_raw(p, CFG["size"])
        msk = brain_mask(arr) if masked else None
        vols[k] = _affine_to(ref_img, arr, iters, msk).astype(np.float32)
        if i%15==0: print(f"  {tag} reg {i}/{len(imgs)} ({time.time()-t0:.0f}s)", flush=True)
    return vols

def trimmed_S(qids, gids, qv, gv, qd, gd, TRIM, symmetric=False):
    n,m=len(qids),len(gids); S=np.zeros((n,m),np.float32)
    for i,qi in enumerate(qids):
        for j,gj in enumerate(gids):
            msk=(qv[qi]>0.05)|(gv[gj]>0.05)
            dvox=np.abs(qd[qi]-gd[gj]).mean(0)[msk]
            dv=np.sort(dvox); k=max(1,int(round((1-TRIM)*dv.size)))
            s=-float(dv[:k].mean()) if dv.size else 0.0
            if symmetric:
                dv2=np.sort(dvox); s=-float(dv2[:k].mean())   # symmetric mask is identical here; kept for clarity
            S[i,j]=s
    return S

def hungarian_rank(S, qids, gids):
    row,col=linear_sum_assignment(-S); assigned={qids[r]:gids[c] for r,c in zip(row,col)}
    out={}
    for i,qid in enumerate(qids):
        agt=assigned[qid]; rest=[gids[j] for j in np.argsort(-S[i]) if gids[j]!=agt]
        out[qid]=[agt]+rest
    return out

def write(path, rows):
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["query_id","target_id_ranking"])
        for qid,r in rows: w.writerow([qid," ".join(r)])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",required=True); a=ap.parse_args()
    root=Path(a.data_root).resolve()
    variants = {f"trim{t}": [] for t in CFG["trims"]}
    variants["mind_trim0.75"]=[]; variants["masked_trim0.75"]=[]
    for split in ["val","test"]:
        qc=read(root/f"dataset3/{split}_queries.csv"); gc=read(root/f"dataset3/{split}_gallery.csv")
        qids=[r["query_id"] for r in qc]; gids=[r["target_id"] for r in gc]
        qimg={r["query_id"]:fix_path(root,r["query_image"]) for r in qc}
        gimg={r["target_id"]:fix_path(root,r["target_image"]) for r in gc}
        print(f"dataset3/{split}: building template + 2 registration passes (standard, brain-masked)...", flush=True)
        tmpl=_sitk(build_template([load_raw(p,CFG["size"]) for p in gimg.values()], CFG["template_iters"]))
        # PASS 1: standard content registration (shared by all trim variants + MIND)
        gv=reg_all(gimg,tmpl,CFG["reg_iters"],f"{split}/gal"); qv=reg_all(qimg,tmpl,CFG["reg_iters"],f"{split}/qry")
        gdS={k:ssc_field(v) for k,v in gv.items()}; qdS={k:ssc_field(v) for k,v in qv.items()}
        gdM={k:mind_field(v) for k,v in gv.items()}; qdM={k:mind_field(v) for k,v in qv.items()}
        # trim sweep (SSC)
        for t in CFG["trims"]:
            S=trimmed_S(qids,gids,qv,gv,qdS,gdS,t); h=hungarian_rank(S,qids,gids)
            for qid in qids: variants[f"trim{t}"].append((qid,h[qid]))
        # MIND at 0.75
        S=trimmed_S(qids,gids,qv,gv,qdM,gdM,0.75); h=hungarian_rank(S,qids,gids)
        for qid in qids: variants["mind_trim0.75"].append((qid,h[qid]))
        # PASS 2: brain-masked registration (SSC, 0.75)
        print(f"dataset3/{split}: brain-masked re-registration...", flush=True)
        gvm=reg_all(gimg,tmpl,CFG["reg_iters"],f"{split}/gal-masked",masked=True)
        qvm=reg_all(qimg,tmpl,CFG["reg_iters"],f"{split}/qry-masked",masked=True)
        gdSm={k:ssc_field(v) for k,v in gvm.items()}; qdSm={k:ssc_field(v) for k,v in qvm.items()}
        S=trimmed_S(qids,gids,qvm,gvm,qdSm,gdSm,0.75); h=hungarian_rank(S,qids,gids)
        for qid in qids: variants["masked_trim0.75"].append((qid,h[qid]))
        print(f"  dataset3/{split} done", flush=True)
    for name,rows in variants.items():
        write(f"/shared-docker/yantra/submission_d3_{name}.csv", rows)
        print(f"wrote submission_d3_{name}.csv ({len(rows)} rows)", flush=True)

if __name__ == "__main__": main()
