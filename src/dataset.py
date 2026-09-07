import os
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

"""
Dataset class for loading ASVspoof audio files and extracting Mel spectrograms
"""
class ASVDataset(Dataset):
    def __init__(self, prot_path: str, aud_dir: str, max_t: int = 400):
        self.data = []
        self.aud_dir = aud_dir
        self.max_t = max_t

        with open(prot_path, 'r') as f:
            for ln in f:
                pts = ln.strip().split()
                if len(pts) < 5:
                    continue
                k = pts[1]
                lbl = 0
                if pts[4] == 'bonafide':
                    lbl = 1
                self.data.append((k, lbl))

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            power=2.0
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        k, y = self.data[i]
        p = os.path.join(self.aud_dir, k + ".flac")

        w, sr = torchaudio.load(p)
        x = self.mel(w)
        x = torch.log(x + 1e-9)

        x = (x - x.mean()) / (x.std() + 1e-7)

        t = x.size(2)
        if t < self.max_t:
            pad = self.max_t - t
            x = F.pad(x, (0, pad))
        else:
            x = x[:, :, :self.max_t]

        return x, y, k