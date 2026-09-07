import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from src.model import LCNN
from src.dataset import ASVDataset
from src.metrics import compute_eer


def parse_args():
    parser = argparse.ArgumentParser(description="Train LCNN Countermeasure on ASVspoof 2019 LA")
    parser.add_argument("--data_dir", type=str, default="data/LA", help="Path to ASVspoof 2019 LA directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_dir = args.data_dir
    tr_prot = os.path.join(base_dir, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
    tr_dir = os.path.join(base_dir, 'ASVspoof2019_LA_train/flac')
    ev_prot = os.path.join(base_dir, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
    ev_dir = os.path.join(base_dir, 'ASVspoof2019_LA_eval/flac')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Запуск обучения на:", dev)

    tr_ds = ASVDataset(tr_prot, tr_dir)
    ev_ds = ASVDataset(ev_prot, ev_dir)

    num_workers = 2 if dev.type == 'cuda' else 0

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    ev_dl = DataLoader(ev_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    mod = LCNN(c_in=1, n_cls=2).to(dev)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(mod.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if not args.no_wandb:
        wandb.init(
            project="asvspoof_lcnn",
            name="lcnn_baseline",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "architecture": "LCNN"
            }
        )

    best_eer = float('inf')

    for ep in range(args.epochs):
        mod.train()
        tr_loss = 0.0
        for x, y, z in tr_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = mod(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        mod.eval()
        ev_loss = 0.0
        res = []
        tgt = []
        ntgt = []

        with torch.no_grad():
            for x, y, k in ev_dl:
                x, y = x.to(dev), y.to(dev)
                out = mod(x)
                loss = crit(out, y)
                ev_loss += loss.item()

                prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                y_np = y.cpu().numpy()

                for i in range(len(k)):
                    res.append((k[i], prob[i]))
                    if y_np[i] == 1:
                        tgt.append(prob[i])
                    else:
                        ntgt.append(prob[i])

        eer, _ = compute_eer(np.array(tgt), np.array(ntgt))

        avg_tr_loss = tr_loss / len(tr_dl)
        avg_ev_loss = ev_loss / len(ev_dl)

        print(
            f"Эпоха {ep + 1}/{args.epochs} | train_loss: {avg_tr_loss:.4f} | eval_loss: {avg_ev_loss:.4f} | EER: {eer * 100:.2f}%")

        if not args.no_wandb:
            wandb.log({
                "epoch": ep + 1,
                "train_loss": avg_tr_loss,
                "eval_loss": avg_ev_loss,
                "eval_eer": eer * 100
            })

        if eer < best_eer:
            best_eer = eer
            model_save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(mod.state_dict(), model_save_path)
            print(f'[*] New best EER: {best_eer * 100:.2f}%. Model saved to {model_save_path}')

            out_file = "predictions.csv"
            with open(out_file, 'w', newline='') as f:
                w = csv.writer(f)
                for k, p in res:
                    w.writerow([k, p])
            print(f"[*] Файл предсказаний сохранен в {out_file}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
