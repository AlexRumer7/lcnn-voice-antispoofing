import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import LCNN
from src.dataset import ASVDataset
from src.metrics import compute_eer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained LCNN on ASVspoof 2019 LA")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth",
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default="data/LA",
                        help="Path to ASVspoof 2019 LA root directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Используемое устройство:", dev)

    eval_prot = os.path.join(args.data_dir, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
    eval_audio = os.path.join(args.data_dir, 'ASVspoof2019_LA_eval/flac')

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Файл весов не найден: {args.model_path}")

    eval_ds = ASVDataset(eval_prot, eval_audio)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=2 if dev.type == 'cuda' else 0)

    model = LCNN(c_in=1, n_cls=2).to(dev)
    model.load_state_dict(torch.load(args.model_path, map_location=dev))
    model.eval()

    tgt = []
    ntgt = []

    print("[*] Запуск оценки...")
    with torch.no_grad():
        for x, y, _ in eval_dl:
            x = x.to(dev)
            out = model(x)
            prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            y_np = y.numpy()

            for p, lbl in zip(prob, y_np):
                if lbl == 1:
                    tgt.append(p)
                else:
                    ntgt.append(p)

    eer, threshold = compute_eer(np.array(tgt), np.array(ntgt))
    print(f"\n[+] Результаты оценки:")
    print(f"    EER: {eer * 100:.4f}%")
    print(f"    Порог решения (Threshold): {threshold:.4f}")


if __name__ == "__main__":
    main()
