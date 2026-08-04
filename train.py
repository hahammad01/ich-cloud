"""
Train one model under the shared config. Run it twice, once per backbone, and the
comparison is fair by construction (same split, epochs, schedule, augmentation).

    python train.py --model resnet50
    python train.py --model efficientnet_b0

Class imbalance (EDH is the rarest) is handled with BCE pos_weight computed from
the TRAIN split only. Best checkpoint is selected on validation macro-AUC.
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as C
from dataset import ICHDataset
from model import build_model


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pos_weight_from_train(df: pd.DataFrame) -> torch.Tensor:
    tr = df[df.split == "train"]
    pos = tr[C.SUBTYPES].sum().values.astype(np.float32)
    neg = len(tr) - pos
    w = np.clip(neg / np.maximum(pos, 1), 1.0, 50.0)   # cap so EDH doesn't explode
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_logits.append(torch.sigmoid(logits).cpu().numpy())
        all_y.append(y.numpy())
    p = np.concatenate(all_logits); y = np.concatenate(all_y)
    aucs = {}
    for i, name in enumerate(C.SUBTYPES):
        aucs[name] = roc_auc_score(y[:, i], p[:, i]) if y[:, i].sum() > 0 else float("nan")
    aucs["macro"] = float(np.nanmean([aucs[s] for s in C.SUBTYPES]))
    return aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["resnet50", "efficientnet_b0", "efficientnet_b3"])
    args = ap.parse_args()

    set_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(C.WORK_DIR / "labels.csv")

    train_ds = ICHDataset(df, "train", train=True)
    val_ds   = ICHDataset(df, "val",   train=False)
    train_dl = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                          num_workers=C.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                          num_workers=C.NUM_WORKERS, pin_memory=True)

    model = build_model(args.model).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from_train(df).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR,
                                  weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=C.USE_AMP)

    best_macro, ckpt = -1.0, C.WORK_DIR / f"best_{args.model}.pt"
    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_dl, desc=f"[{args.model}] epoch {epoch}/{C.EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=C.USE_AMP):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running += loss.item()
        scheduler.step()

        aucs = evaluate(model, val_dl, device)
        print(f"  loss {running/len(train_dl):.4f} | macro-AUC {aucs['macro']:.4f} | "
              f"EDH {aucs['epidural']:.4f}")
        if aucs["macro"] > best_macro:
            best_macro = aucs["macro"]
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "val_auc": aucs}, ckpt)
            print(f"  saved -> {ckpt}")

    print(f"Best val macro-AUC ({args.model}): {best_macro:.4f}")


if __name__ == "__main__":
    main()
