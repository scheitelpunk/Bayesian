# ⚡ GPU Fix - Sofort-Anleitung

## Problem
Du hast PyTorch **CPU-only** Version installiert: `2.9.0+cpu`

Deine RTX 5060 wird **nicht genutzt**, obwohl CUDA 13.0 verfügbar ist!

---

## ✅ Lösung (5 Minuten)

### Schritt 1: Alte PyTorch-Version deinstallieren
```bash
pip uninstall torch torchvision torchaudio -y
```

### Schritt 2: PyTorch mit CUDA 12.4 installieren
(CUDA 13.0 wird noch nicht von PyTorch unterstützt, nutze 12.4 - ist kompatibel!)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Alternative für CUDA 12.1** (falls 12.4 nicht verfügbar):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Schritt 3: Verifizieren
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Erwartete Ausgabe**:
```
CUDA: True
GPU: NVIDIA GeForce RTX 5060
```

### Schritt 4: Training mit GPU
```bash
python examples/real_data_demo.py
```

**Jetzt sollte angezeigt werden**:
```
Device: cuda:0
```

---

## 🚀 Erwarteter Performance-Boost

| Vorher (CPU) | Nachher (GPU RTX 5060) |
|--------------|------------------------|
| ~30-60 Sekunden | **~5-10 Sekunden** |
| batch_size=8 | **batch_size=64** möglich |
| 1000 samples | **25,000 samples** trainierbar |

**→ 6-12x schneller!** ⚡

---

## 📊 Nach GPU-Setup: Optimierungen

### 1. Größere Batch Size
```python
# In examples/real_data_demo.py
batch_size = 32  # oder sogar 64!
```

### 2. Mehr Trainingsdaten
```python
max_samples_train = 25000  # statt 1000
max_samples_test = 5000    # statt 200
```

### 3. Mixed Precision (2x schneller)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
```

---

## 🔍 Troubleshooting

**Falls "CUDA out of memory"**:
- Reduziere `batch_size` auf 16
- Oder nutze Gradient Checkpointing

**Falls GPU nicht erkannt**:
- Prüfe mit `nvidia-smi` ob GPU sichtbar
- Installiere neueste NVIDIA Treiber

---

**Das war's! Nach Installation hast du 6-12x schnelleres Training!** 🚀
