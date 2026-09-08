# Voice Anti-Spoofing & Deepfake Detection with Light CNN 

A PyTorch implementation of a Countermeasure system for acoustic deepfake detection and logical access anti-spoofing, evaluated on the **ASVspoof 2019 Logical Access** dataset.

The system is designed to distinguish bona fide human speech from synthesized and voice-converted audio attacks using a **Lightweight Convolutional Neural Network (LCNN)** architecture powered by **Max-Feature-Map (MFM)** activations.

---

## Architecture Overview

```
 Raw Audio 
   │
   ▼
  [Log Mel-Spectrogram (80 mels, n_fft=1024, hop=256)]
   │
   ▼
 [ Conv1: 5x5, 64 ch ] ──► [ MFM: 32 ch ] ──► [ MaxPool 2x2 ] 
   │
   ▼
 [ Conv2: 1x1, 64 ch ] ──► [ MFM: 32 ch ] ──► [ BatchNorm ] 
   │
   ▼
 [ Conv2: 3x3, 96 ch ] ──► [ MFM: 48 ch ] ──► [ MaxPool 2x2 ] ──► [ BatchNorm ] 
   │
   ▼
 [ Conv3: 1x1, 96 ch ] ──► [ MFM: 48 ch ] ──► [ BatchNorm ] ──► [ Conv3: 3x3, 128 ch] ──► [ MFM: 64 ch ] ──► [ MaxPool 2x2 ] 
   │
   ▼
 [ Conv4: 1x1, 128 ch] ──► [ MFM: 64 ch ] ──► [ BatchNorm ] ──► [ Conv4: 3x3, 64 ch ] ──► [ MFM: 32 ch ] ──► [ BatchNorm ] ──► [ Conv4: 3x3, 64 ch ] ──► [ MFM: 32 ch ] ──► [ MaxPool 2x2 ] 
   │
   ▼
 [ AdaptiveAvgPool2d (1x1)] 
   │
   ▼
 [ Flatten (32 dims) ] 
   │
   ▼
 [ Dropout (p = 0.5) ] 
   │
   ▼
 [ BatchNorm1d (32 dims) ] 
   │
   ▼
 [ Linear (32 -> 2 cls) ]

```

### Key Technical Details
- **Acoustic Front-end**: 80-channel Log Mel-Filterbanks extracted with torchaudio. Features are normalized per utterance and padded/truncated to 400 temporal frames (~6.4 seconds).
- **Backbone**: Light CNN with MFM non-linearities. MFM acts as a competitive activation function that splits feature channels and takes the element-wise maximum.
- **Regularization**: Dropout ($p=0.5$) applied right before final batch normalization and classification head to prevent overfitting on unseen attack types.
- **Objective Function**: Cross-Entropy Loss optimized with Adam.
- **Evaluation Metrics**: Equal Error Rate (EER).

---

## Repository Structure

```text
├── src/
│   ├── dataset.py      # ASVspoof dataset loader, preprocessing, and mel extraction
│   ├── model.py        # LCNN architecture and MFM layer
│   └── metrics.py      # EER computation tools
├── train.py            # Model training pipeline with WandB integration
├── evaluate.py         # Checkpoint evaluation script on the evaluation set
├── requirements.txt    # Project dependencies
└── README.md
```

---

## Getting Started


Python 3.9+ and PyTorch with CUDA support are recommended:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Training

To train the LCNN countermeasure from scratch:

```bash
python train.py \
    --data_dir data/LA \
    --epochs 15 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.0001 \
    --output_dir checkpoints
```

During training, the script evaluates the model on each epoch, saves the best checkpoint (`best_model.pth`) based on the lowest EER, and exports soft prediction scores to `predictions.csv`.

---

## Evaluation

To evaluate a trained checkpoint on the ASVspoof 2019 LA evaluation set:

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_dir data/LA \
    --batch_size 32
```

---

## Benchmark Results

Evaluated on the **ASVspoof 2019 LA Evaluation Set**:

| Architecture | Front-end | Loss | EER     |
|:-------------| :--- | :--- |:--------|
| **LCNN**     | **80 Log Mel** | **Cross-Entropy** | **4.9** |

(Trained for 15 epochs; batch size 32, Adam optimizer with initial lr=1e-4)

---