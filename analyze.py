"""
Analyse a saved predictions file (work/test_preds_<model>.npz) written by
evaluate.py. CPU-only -- no GPU, no retraining -- so it runs even with zero quota.

    python analyze.py --npz work/test_preds_efficientnet_b0.npz
    python analyze.py --npz work/test_preds_efficientnet_b0.npz \
                      --vs work/test_preds_resnet50.npz          # side-by-side

What it produces (in work/analysis/):
  - printed report: ROC-AUC, PR-AUC, calibration, operating points, prevalence
    projections, and the EDH false-alarm framing
  - pr_curves_<model>.png, calibration_<model>.png
  - metrics_<model>.csv

METHOD NOTE printed at the top: threshold-independent metrics (ROC-AUC, PR-AUC)
are the honest headline. Threshold-dependent numbers are reported as operating
points at FIXED target sensitivities, read off the curve -- NOT by optimising a
threshold on the test set (that would inflate the result). Re-select any single
"chosen" threshold on the validation split before final reporting.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless: save figures, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve, brier_score_loss)

TARGET_SENS = [0.80, 0.90, 0.95]           # operating points to report
PROJECT_PREV = [0.005, 0.01, 0.02, 0.05]   # prevalences for PPV/NPV projection
OUT_DIR = Path("work/analysis")


# --------------------------------------------------------------------------
def load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d["probs"], d["labels"], [str(s) for s in d["subtypes"]]


def metrics_at_threshold(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return dict(threshold=t, sens=sens, spec=spec, ppv=ppv, npv=npv,
                tp=tp, fp=fp, fn=fn, tn=tn)


def threshold_for_sensitivity(y, p, target):
    """Highest threshold whose sensitivity (TPR) is still >= target."""
    fpr, tpr, thr = roc_curve(y, p)
    hits = np.where(tpr >= target)[0]
    if len(hits) == 0:
        return 0.0
    return float(thr[hits[0]])           # roc thresholds are descending


def ece(y, p, n_bins=10):
    """Expected Calibration Error."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return e


def project_ppv_npv(sens, spec, prev):
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev) + 1e-12)
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev + 1e-12)
    return ppv, npv


# --------------------------------------------------------------------------
def analyse(npz_path):
    probs, labels, subtypes = load(npz_path)
    model = Path(npz_path).stem.replace("test_preds_", "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"ANALYSIS: {model}   (n = {len(probs)} slices)")
    print("Headline metrics are threshold-INDEPENDENT (ROC-AUC, PR-AUC).")
    print("Operating points are read at FIXED target sensitivities, not tuned on test.")
    print("=" * 74)

    rows = []
    for i, name in enumerate(subtypes):
        y, p = labels[:, i].astype(int), probs[:, i]
        if y.sum() == 0:
            print(f"\n{name}: no positives, skipped"); continue

        prev = y.mean()
        roc = roc_auc_score(y, p)
        pr = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        cal = ece(y, p)

        print(f"\n{'-'*74}\n{name.upper()}   "
              f"(positives={int(y.sum())}, test prevalence={prev:.3%})")
        print(f"  ROC-AUC {roc:.3f} | PR-AUC {pr:.3f} | Brier {brier:.4f} | ECE {cal:.4f}")

        row = dict(model=model, subtype=name, n_pos=int(y.sum()),
                   test_prevalence=prev, roc_auc=roc, pr_auc=pr,
                   brier=brier, ece=cal)

        print(f"  {'target sens':>12}{'threshold':>11}{'PPV':>8}{'spec':>8}"
              f"{'FP/TP':>8}")
        for ts in TARGET_SENS:
            t = threshold_for_sensitivity(y, p, ts)
            m = metrics_at_threshold(y, p, t)
            fp_per_tp = (m["fp"] / m["tp"]) if m["tp"] else float("inf")
            print(f"  {ts:>12.2f}{t:>11.3f}{m['ppv']:>8.3f}{m['spec']:>8.3f}"
                  f"{fp_per_tp:>8.1f}")
            row[f"ppv@sens{int(ts*100)}"] = m["ppv"]
            row[f"thr@sens{int(ts*100)}"] = t
            if abs(ts - 0.90) < 1e-9:          # keep the 0.90 point for projection
                sens90, spec90 = m["sens"], m["spec"]

        # prevalence-projected PPV/NPV at the sens>=0.90 operating point
        print(f"  projected PPV / NPV at sens=0.90 (spec={spec90:.3f}) across prevalence:")
        for pv in PROJECT_PREV:
            ppv_p, npv_p = project_ppv_npv(sens90, spec90, pv)
            print(f"      prevalence {pv:>6.1%}:  PPV {ppv_p:.3f}   NPV {npv_p:.4f}")
            row[f"proj_ppv@prev{pv}"] = ppv_p

        rows.append(row)

    # ---- EDH spotlight -------------------------------------------------
    if "epidural" in subtypes:
        i = subtypes.index("epidural")
        y, p = labels[:, i].astype(int), probs[:, i]
        t = threshold_for_sensitivity(y, p, 0.90)
        m = metrics_at_threshold(y, p, t)
        print(f"\n{'='*74}\nEDH SPOTLIGHT (the project's headline failure mode)")
        if m["tp"]:
            print(f"  At sensitivity 0.90: PPV {m['ppv']:.3f}  ->  about "
                  f"{m['fp']/m['tp']:.0f} false alarms per true EDH.")
        print(f"  Interpret with the prevalence projection above: at realistic ED")
        print(f"  EDH prevalence, PPV is lower still. This gap, not the ROC-AUC, is")
        print(f"  the clinically meaningful finding.")

    # ---- figures -------------------------------------------------------
    _pr_plot(probs, labels, subtypes, model)
    _cal_plot(probs, labels, subtypes, model)

    df = pd.DataFrame(rows)
    csv = OUT_DIR / f"metrics_{model}.csv"
    df.to_csv(csv, index=False)
    print(f"\nSaved: {csv}")
    print(f"Saved: {OUT_DIR/f'pr_curves_{model}.png'}")
    print(f"Saved: {OUT_DIR/f'calibration_{model}.png'}")
    return df


def _pr_plot(probs, labels, subtypes, model):
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(subtypes):
        y, p = labels[:, i].astype(int), probs[:, i]
        if y.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall (sensitivity)"); plt.ylabel("Precision (PPV)")
    plt.title(f"Precision-Recall by subtype — {model}")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / f"pr_curves_{model}.png", dpi=130)
    plt.close()


def _cal_plot(probs, labels, subtypes, model, n_bins=10):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for ax, i in zip(axes.ravel(), range(len(subtypes))):
        y, p = labels[:, i].astype(int), probs[:, i]
        idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        acc = [y[idx == b].mean() if (idx == b).any() else np.nan for b in range(n_bins)]
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.plot(centers, acc, "o-", ms=3)
        ax.set_title(subtypes[i], fontsize=9)
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("observed", fontsize=8)
    fig.suptitle(f"Calibration by subtype — {model}")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"calibration_{model}.png", dpi=130)
    plt.close()


def compare(df_a, df_b):
    print(f"\n{'='*74}\nMODEL COMPARISON (ROC-AUC | PR-AUC)")
    m = df_a.merge(df_b, on="subtype", suffixes=("_a", "_b"))
    print(f"{'subtype':<17}{'model A':>22}{'model B':>22}")
    for _, r in m.iterrows():
        print(f"{r['subtype']:<17}"
              f"{r['roc_auc_a']:>10.3f}/{r['pr_auc_a']:<10.3f}"
              f"{r['roc_auc_b']:>10.3f}/{r['pr_auc_b']:<10.3f}")
    print("A =", df_a['model'].iloc[0], " B =", df_b['model'].iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="predictions .npz to analyse")
    ap.add_argument("--vs", default=None, help="optional second .npz to compare")
    args = ap.parse_args()

    df_a = analyse(args.npz)
    if args.vs:
        df_b = analyse(args.vs)
        compare(df_a, df_b)


if __name__ == "__main__":
    main()
