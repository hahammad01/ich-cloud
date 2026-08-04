"""
Honest test-set evaluation -- this is where your actual contribution lives.

For each subtype it reports ROC-AUC AND PR-AUC (average precision), plus
sensitivity and PPV at a threshold. PR-AUC and PPV matter far more than a headline
AUC for rare classes like EDH: a model can have a great AUC and still fire ~20
false alarms per true EDH. Report both, and disclose the threshold.

    python evaluate.py --model efficientnet_b0 --threshold 0.5
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

import config as C
from dataset import ICHDataset
from model import build_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="single global threshold; NOTE: tune per-subtype later")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(C.WORK_DIR / "labels.csv")
    test_dl = DataLoader(ICHDataset(df, "test", train=False),
                         batch_size=C.BATCH_SIZE, shuffle=False,
                         num_workers=C.NUM_WORKERS, pin_memory=True)

    ckpt = torch.load(C.WORK_DIR / f"best_{args.model}.pt", map_location=device)
    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    p, y = predict(model, test_dl, device)
    t = args.threshold

    print(f"\nTest metrics @ threshold={t}  (model: {args.model})")
    print(f"{'subtype':<17}{'ROC-AUC':>9}{'PR-AUC':>9}{'Sens':>8}{'PPV':>8}{'#pos':>7}")
    for i, name in enumerate(C.SUBTYPES):
        yi, pi = y[:, i], p[:, i]
        if yi.sum() == 0:
            print(f"{name:<17}{'n/a':>9}"); continue
        pred = (pi >= t).astype(int)
        tp = int(((pred == 1) & (yi == 1)).sum())
        fp = int(((pred == 1) & (yi == 0)).sum())
        fn = int(((pred == 0) & (yi == 1)).sum())
        sens = tp / (tp + fn + 1e-9)
        ppv  = tp / (tp + fp + 1e-9)
        print(f"{name:<17}{roc_auc_score(yi, pi):>9.3f}"
              f"{average_precision_score(yi, pi):>9.3f}"
              f"{sens:>8.3f}{ppv:>8.3f}{int(yi.sum()):>7}")

    # Save raw probabilities so you can make PR curves / calibration plots later.
    out = C.WORK_DIR / f"test_preds_{args.model}.npz"
    np.savez(out, probs=p, labels=y, subtypes=np.array(C.SUBTYPES))
    print(f"\nSaved probabilities -> {out}  (use for PR curves, calibration, etc.)")


if __name__ == "__main__":
    main()
