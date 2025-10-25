# ⚠️ RTX 5060 sm_120 - Aktuell nicht unterstützt

## Problem (Stand: Oktober 2025)

**RTX 5060 ist ZU NEU für PyTorch!**

```
RTX 5060 Laptop: CUDA Capability sm_120
PyTorch 2.9.0: Unterstützt nur bis sm_90
```

PyTorch empfiehlt **CUDA 12.8 oder 13.0**, die es noch nicht gibt!

---

## ⚡ 3 Lösungen

### Option 1: **CPU nutzen (EMPFOHLEN bis PyTorch Update)**

```bash
# CPU-Version installieren
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio
```

**Pro**:
- ✅ Funktioniert SOFORT
- ✅ Kein Compatibility-Problem
- ✅ Kein Code-Änderung nötig

**Con**:
- ❌ 6-12x langsamer als GPU
- ❌ Kleinere Batch Sizes

**Training Zeit (1000 samples, 5 epochs)**:
- CPU: ~30-60 Sekunden ✅ (akzeptabel für Tests)
- GPU: ~5-10 Sekunden (wenn verfügbar)

---

### Option 2: **PyTorch Nightly Build (Experimentell)**

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

**Pro**:
- ✅ Könnte sm_120 unterstützen
- ✅ Neueste Features

**Con**:
- ❌ Instabil (Nightly Build!)
- ❌ Bugs möglich
- ❌ Nicht für Production

**Teste mit**:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

### Option 3: **Warten auf offizielles Update**

PyTorch wird sm_120 Support in Zukunft hinzufügen.

**Checke regelmäßig**:
- https://pytorch.org/get-started/locally/
- https://github.com/pytorch/pytorch/releases

**Erwartetes Update**: Q1/Q2 2026

---

## ✅ Empfohlene Lösung: CPU + Optimierungen

### 1. CPU-Version installieren
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio
```

### 2. Code-Optimierungen für CPU

#### A. **Kleinere Batch Size**
```python
# In examples/real_data_demo.py
batch_size = 4  # statt 8 (weniger RAM)
```

#### B. **Weniger Parameter** (für schnelleres Training)
```python
config = {
    'd_model': 256,      # statt 512
    'n_heads': 4,        # statt 8
    'vocab_size': 5000,  # statt 10000
    'k_permutations': 10,  # statt 20
}
```

**Effekt**: 4.2M → ~1.2M Parameter = **3-4x schneller** auf CPU

#### C. **Parallele DataLoader**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    num_workers=4,  # Nutze alle CPU-Kerne
    pin_memory=False  # CPU-only
)
```

#### D. **JIT Compilation** (Optional)
```python
import torch

# Nach Model-Definition
model = torch.jit.script(model)  # Kompiliere für CPU
```

---

## 📊 CPU Performance-Erwartungen

### Mit Optimierungen:

| Dataset Size | Training Time (CPU) | Training Time (ideal GPU) |
|--------------|---------------------|---------------------------|
| 1,000 samples | ~20-30 Sekunden | ~5 Sekunden |
| 5,000 samples | ~2-3 Minuten | ~20-30 Sekunden |
| 20,000 samples | ~10-15 Minuten | ~2-3 Minuten |

**CPU ist vollkommen akzeptabel** für:
- ✅ Development & Testing
- ✅ Kleine Datasets (<5K samples)
- ✅ Proof of Concept
- ✅ Hyperparameter Tuning

---

## 🔄 Wann GPU-Check wiederholen

### Monatlich prüfen ob PyTorch Update verfügbar:
```bash
# Neueste PyTorch Version checken
pip search torch | grep "^torch " || \
curl -s https://pypi.org/pypi/torch/json | grep '"version"'

# Oder PyTorch Website besuchen:
# https://pytorch.org/get-started/locally/
```

### Wenn neues PyTorch verfügbar:
1. Lies Changelog: Support für sm_120?
2. Installiere neue Version
3. Teste GPU:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 🎯 Aktueller Workaround: CPU nutzen

```bash
# Installation
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio

# Training starten (nutzt automatisch CPU)
python examples/real_data_demo.py
```

**Ausgabe sollte zeigen**:
```
Device: cpu
```

**Das ist OK!** CPU ist schnell genug für Development.

---

## 📝 Update dieses Dokuments

Wenn PyTorch sm_120 unterstützt:
1. Update dieses Dokument
2. Installiere GPU-Version: `pip install torch --index-url https://download.pytorch.org/whl/cu130` (oder neuere CUDA)
3. Profit! 6-12x schneller

---

## 💡 Alternative: Google Colab (Kostenlos GPU)

Falls du GPU JETZT brauchst:

### Google Colab (Gratis T4 GPU)
1. Gehe zu https://colab.research.google.com/
2. Upload dein Notebook
3. Runtime → Change runtime type → GPU (T4)
4. Nutze deren GPU kostenlos!

**Pro**:
- ✅ Kostenlose GPU
- ✅ Keine Installation
- ✅ sm_80 wird unterstützt

**Con**:
- ❌ Zeitlimit (12h Sessions)
- ❌ Langsamer als lokale RTX 5060
- ❌ Braucht Internet

---

## ✅ Zusammenfassung

**Für JETZT**: Nutze CPU + Optimierungen
- Installiere CPU PyTorch
- Nutze kleineres Modell (256 dim, 4 heads)
- Training ist akzeptabel schnell (~30 Sek für 1K samples)

**Für SPÄTER**: Warte auf PyTorch Update
- Checke monatlich für sm_120 Support
- Dann: GPU ist 6-12x schneller
- Bis dahin: CPU funktioniert!

**Dein Training läuft trotzdem!** 🚀 Nur etwas langsamer.
