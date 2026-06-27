"""Train Branch A: shared 3D contrastive encoder with InfoNCE + domain-randomization aug.
Oracle-gated, W&B-logged, checkpoints the best oracle-mean.

Usage (box with data):
  python train.py --data-root <root> --train-csv <root>/dataset1/train_pairs.csv \
    --backbone resnet18 --epochs 50 --batch-size 32 --eval-every 5
Quick smoke run: add  --limit-pairs 80 --n-holdout 30 --epochs 3 --eval-every 1
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import CFG
import preprocess
import augment
import oracle
from metrics import mrr


def info_nce_loss(q_emb: torch.Tensor, t_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch negatives (q_emb/t_emb: [B,D] L2-normed)."""
    logits = (q_emb @ t_emb.t()) / temperature
    labels = torch.arange(len(q_emb), device=q_emb.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class PairDataset(Dataset):
    """Positive (query, target) pairs; augment.train_transforms applied INDEPENDENTLY to each."""
    def __init__(self, pairs, id2vol, tf):
        self.pairs, self.id2vol, self.tf = pairs, id2vol, tf

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _plain(x):  # strip MONAI MetaTensor -> plain torch.Tensor (avoids MetaTensor codepaths)
        x = x.as_tensor() if hasattr(x, "as_tensor") else torch.as_tensor(x)
        return x.float()

    def __getitem__(self, i):
        qid, tid = self.pairs[i]
        q = self.tf({"image": self.id2vol[qid].clone()})["image"]
        t = self.tf({"image": self.id2vol[tid].clone()})["image"]
        return self._plain(q), self._plain(t)


@torch.no_grad()
def eval_oracle(enc, holdout, dev, levels=oracle.LEVELS) -> dict:
    """Per-level MRR with the encoder (gallery embeddings precomputed once per level). No align."""
    enc.eval()
    out = {}
    for level in levels:
        try:
            qv, gv = oracle.build_level(holdout, level, CFG)
        except Exception as e:
            out[level] = None
            print(f"  [oracle] skip {level}: {e}")
            continue
        qe = torch.stack([enc.encode(v.to(dev)) for v in qv])      # [Q,d]
        ge = torch.stack([enc.encode(v.to(dev)) for v in gv])      # [G,d]
        S = qe @ ge.t()                                            # cosine (unit-norm) [Q,G]
        rankings = {i: torch.argsort(S[i], descending=True).tolist() for i in range(len(qv))}
        out[level] = mrr(rankings, {i: i for i in range(len(qv))})
    done = [out[l] for l in levels if out.get(l) is not None]
    out["mean"] = sum(done) / len(done) if done else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=CFG.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--resolution", type=int, default=CFG.resolution)
    ap.add_argument("--n-holdout", type=int, default=CFG.holdout_pairs)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--limit-pairs", type=int, default=0, help=">0 = subset for a quick smoke run")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="outputs_gs/encoder_best.pt")
    ap.add_argument("--wandb-project", default="ehl-mri-retrieval")
    ap.add_argument("--run-name", default="encoder-infonce")
    args = ap.parse_args()

    CFG.data_root = Path(args.data_root)
    CFG.resolution = args.resolution
    dev = torch.device(args.device)

    rows = list(csv.DictReader(open(args.train_csv, newline="")))
    pairs_meta = [{"query_id": r["query_id"], "target_id": r["target_id"],
                   "query_image": r["query_image"], "target_image": r["target_image"]} for r in rows]
    _, hold_meta = oracle.make_holdout(pairs_meta, n=args.n_holdout)
    hold_ids = {p["query_id"] for p in hold_meta} | {p["target_id"] for p in hold_meta}
    train_meta = [p for p in pairs_meta if p["query_id"] not in hold_ids and p["target_id"] not in hold_ids]
    if args.limit_pairs:
        train_meta = train_meta[:args.limit_pairs]
    print(f"pairs: {len(pairs_meta)} | train {len(train_meta)} | holdout {len(hold_meta)} | dev {dev}")

    # preprocess every needed image ONCE into memory (CPU tensors)
    need = {}
    for p in train_meta + hold_meta:
        need[p["query_id"]] = p["query_image"]
        need[p["target_id"]] = p["target_image"]
    t0 = time.time()
    id2vol = {i: preprocess.preprocess_volume(path) for i, path in need.items()}
    print(f"preprocessed {len(id2vol)} volumes in {time.time() - t0:.1f}s")
    holdout = [{"query": id2vol[p["query_id"]], "target": id2vol[p["target_id"]]} for p in hold_meta]

    tf = augment.train_transforms(CFG)
    ds = PairDataset([(p["query_id"], p["target_id"]) for p in train_meta], id2vol, tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.num_workers)

    from encoder import Encoder
    enc = Encoder(CFG, backbone=args.backbone).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG.amp and dev.type == "cuda"))

    wb = None
    if args.wandb_project:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=args.run_name,
                            config={"backbone": args.backbone, "epochs": args.epochs,
                                    "batch_size": args.batch_size, "lr": args.lr,
                                    "resolution": args.resolution, "temperature": CFG.temperature,
                                    "n_train": len(train_meta)}, settings=wandb.Settings(silent=True))
            print("wandb:", wb.get_url())
        except Exception as e:
            print("wandb disabled:", e)

    best = -1.0
    step = 0
    for epoch in range(1, args.epochs + 1):
        enc.train()
        tot, n = 0.0, 0
        for q, t in dl:
            q, t = q.to(dev), t.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=(CFG.amp and dev.type == "cuda")):
                qe, te = enc(q), enc(t)
                loss = info_nce_loss(qe, te, CFG.temperature)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(q); n += len(q); step += 1
            if wb:
                wb.log({"train/loss": float(loss)}, step=step)
        print(f"epoch {epoch:03d} loss={tot / max(n, 1):.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            res = eval_oracle(enc, holdout, dev)
            print(f"  oracle: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in res.items()))
            if wb:
                wb.log({f"oracle/{k}": v for k, v in res.items() if isinstance(v, float)}, step=step)
            if res["mean"] > best:
                best = res["mean"]
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": enc.state_dict(), "backbone": args.backbone,
                            "oracle": res, "epoch": epoch}, args.out)
                print(f"  saved best (mean={best:.4f}) -> {args.out}")
    if wb:
        wb.finish()
    print(f"DONE best oracle mean={best:.4f}")


if __name__ == "__main__":
    main()
