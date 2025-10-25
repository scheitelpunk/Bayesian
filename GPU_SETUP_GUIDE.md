# 🚀 GPU Setup Guide - NVIDIA RTX 5060

## ❌ Problem: PyTorch erkennt GPU nicht

**Aktueller Status**: `CUDA Available: False`
**Deine Hardware**: NVIDIA RTX 5060 (8GB)

---

## 🔧 Lösung: PyTorch mit CUDA neu installieren

### Schritt 1: NVIDIA Treiber prüfen

```bash
# GPU Status prüfen
nvidia-smi

# Erwartete Ausgabe:
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 5xx.xx       Driver Version: 5xx.xx         CUDA Version: 12.x             |
# +-----------------------------------------------------------------------------------------+
# | GPU  Name                  TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC    |
# | Fan  Temp  Perf          Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M.   |
# +-----------------------------------------------------------------------------------------+
# |   0  NVIDIA GeForce RTX 5060   On  | 00000000:01:00.0  On |                  N/A    |
```

**Falls nvidia-smi nicht funktioniert**:
1. Lade neueste NVIDIA Treiber: https://www.nvidia.com/Download/index.aspx
2. Wähle: RTX 5060, Windows 11, Game Ready Driver
3. Installiere und restarte

---

### Schritt 2: CUDA Version identifizieren

```bash
# CUDA Version aus nvidia-smi ablesen
nvidia-smi

# Beispiel: "CUDA Version: 12.1" → Nutze CUDA 12.1
```

---

### Schritt 3: PyTorch mit CUDA neu installieren

**WICHTIG**: Deinstalliere erst die CPU-Version!

```bash
# 1. Alte PyTorch-Version deinstallieren
pip uninstall torch torchvision torchaudio -y

# 2. PyTorch mit CUDA 12.1 installieren (empfohlen für RTX 5060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Oder CUDA 11.8 (falls 12.1 Probleme macht)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Alternative**: Besuche https://pytorch.org/get-started/locally/ und wähle:
- PyTorch Build: Stable
- Your OS: Windows
- Package: Pip
- Language: Python
- Compute Platform: CUDA 12.1 (oder passende Version)

---

### Schritt 4: GPU-Support verifizieren

```bash
# Test ob CUDA funktioniert
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Erwartete Ausgabe:
# CUDA Available: True
# GPU Name: NVIDIA GeForce RTX 5060
```

---

## 🚀 Training mit GPU starten

Nach erfolgreicher Installation:

```bash
# Automatische GPU-Nutzung
python examples/real_data_demo.py

# Ausgabe sollte zeigen:
# Device: cuda:0
# (statt Device: cpu)
```

Der Code erkennt **automatisch** die GPU:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## ⚡ Erwarteter Performance-Boost

| Metrik | CPU (jetzt) | GPU (RTX 5060) | Speedup |
|--------|-------------|----------------|---------|
| Training (1000 samples, 5 epochs) | ~30-60 Sekunden | ~5-10 Sekunden | **6-12x schneller** |
| Inference (single prediction) | ~50ms | ~5-10ms | **5-10x schneller** |
| Batch Inference (32 samples) | ~350ms | ~50-80ms | **4-7x schneller** |
| Memory | 4-8GB RAM | 2-4GB VRAM | Mehr verfügbar für größere Batches |

---

## 🎯 Optimierungen nach GPU-Setup

### 1. Größere Batch Size nutzen

```python
# In real_data_demo.py
batch_size = 32  # Aktuell: 8
# Mit 8GB VRAM kannst du bis zu batch_size=64 nutzen!
```

**Effekt**: 2-3x schnelleres Training durch bessere GPU-Auslastung

---

### 2. Mehr Daten trainieren

```python
# In real_data_demo.py, Zeile ~400
max_samples_train = 25000  # Aktuell: 1000
max_samples_test = 5000    # Aktuell: 200
```

**Effekt**: Bessere Modell-Genauigkeit, nur ~5-10 Minuten Training (statt Stunden auf CPU)

---

### 3. Mixed Precision Training (AMP)

```python
# In real_data_demo.py, train_epoch() Funktion
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Effekt**: 2x schneller + 50% weniger VRAM durch FP16

---

### 4. DataLoader Optimierung

```python
# In real_data_demo.py
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=4,      # Aktuell: 0
    pin_memory=True,    # Schnellerer CPU→GPU Transfer
    persistent_workers=True
)
```

**Effekt**: Überlappt Daten-Loading mit GPU-Berechnung → 20-30% schneller

---

## 🔍 Troubleshooting

### Problem: "CUDA out of memory"

**Lösung**:
```python
# Reduziere batch_size
batch_size = 16  # statt 32

# Oder nutze Gradient Checkpointing
torch.utils.checkpoint.checkpoint(model, x)
```

### Problem: "RuntimeError: CUDA error: no kernel image is available"

**Lösung**: Falsche CUDA-Version installiert
```bash
# Prüfe CUDA-Kompatibilität
python -c "import torch; print(torch.version.cuda)"

# Sollte mit nvidia-smi CUDA Version übereinstimmen
```

### Problem: GPU wird nicht genutzt trotz CUDA Available: True

**Lösung**: Code nutzt explizit CPU
```python
# Prüfe in real_data_demo.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model MUSS auf GPU verschoben werden
model = model.to(device)
```

---

## 📊 GPU Memory Management

### Aktueller Memory Verbrauch prüfen

```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
```

### Memory freigeben

```python
# Nach Training
torch.cuda.empty_cache()
```

---

## ✅ Checkliste: GPU Setup erfolgreich

- [ ] `nvidia-smi` zeigt GPU an
- [ ] `torch.cuda.is_available()` gibt `True` zurück
- [ ] `torch.cuda.get_device_name(0)` zeigt "RTX 5060"
- [ ] Training zeigt `Device: cuda:0`
- [ ] GPU-Auslastung sichtbar in nvidia-smi (während Training)
- [ ] Training ist 6-12x schneller als vorher

---

## 🚀 Next Level: Multi-GPU (falls du später upgradest)

```python
# Für 2+ GPUs: DataParallel
model = nn.DataParallel(model)

# Oder besser: DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[0, 1])
```

---

## 📚 Weiterführende Ressourcen

- **PyTorch CUDA Setup**: https://pytorch.org/get-started/locally/
- **NVIDIA Treiber**: https://www.nvidia.com/Download/index.aspx
- **CUDA Toolkit** (optional): https://developer.nvidia.com/cuda-downloads
- **Mixed Precision Guide**: https://pytorch.org/docs/stable/amp.html
- **Performance Tuning**: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

---

**Nach GPU-Setup**: Dein Training wird **6-12x schneller** und du kannst mit dem vollen IMDB-Dataset (25K samples) arbeiten! 🚀
