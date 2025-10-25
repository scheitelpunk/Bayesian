# 🔍 Analyse: Training Results & Overfitting

## ⚠️ Problem erkannt: Perfektes Overfitting!

### Ergebnisse der letzten Demo-Ausführung:

```
Epoch 1/5: Loss = 0.0505, Accuracy = 0.9940
Epoch 2/5: Loss = 0.0002, Accuracy = 1.0000
Epoch 3/5: Loss = 0.0001, Accuracy = 1.0000
Epoch 4/5: Loss = 0.0000, Accuracy = 1.0000
Epoch 5/5: Loss = 0.0000, Accuracy = 1.0000

Test Accuracy: 1.0000 (100%)
Uncertainty: 0.0556 (alle Samples fast identisch!)
```

---

## 🚨 Was ist falsch?

### 1. **Extremes Overfitting**
- Training Accuracy: **100%** ❌
- Validation Accuracy: **100%** ❌
- Loss: **0.0000** ❌ (völlig unrealistisch!)

**Warum schlecht?**
- Modell hat Trainingsdaten **auswendig gelernt**
- Wird auf neuen Daten **sehr schlecht** generalisieren
- Uncertainty ist nutzlos (alle ~0.0556)

### 2. **Zu kleine Dataset**
- **1000 Trainingssamples** für 4.2M Parameter
- **200 Test samples** (viel zu wenig für Validierung)

**Ratio**: 4,234,801 Parameter / 1000 Samples = **4,234 Parameter pro Sample!**

Gesunde Ratio: ~10-100 Samples pro Parameter

### 3. **Keine echte Uncertainty**
- Alle Uncertainties: 0.0556 ± 0.0001
- Bedeutet: Modell ist **immer 100% sicher** (red flag!)
- Bayesian Uncertainty funktioniert nicht

---

## ✅ Empfehlungen zur Behebung

### 🎯 Sofort (Quick Wins)

#### 1. **Mehr Trainingsdaten nutzen**

```python
# In examples/real_data_demo.py, Zeile ~400
# VORHER:
max_samples_train = 1000
max_samples_test = 200

# NACHHER:
max_samples_train = 20000  # 20x mehr!
max_samples_test = 5000    # 25x mehr!
```

**Erwartung**:
- Training Accuracy: 85-92% (realistisch)
- Validation Accuracy: 80-88%
- Uncertainties: 0.05-0.30 (variabel!)

---

#### 2. **Regularization hinzufügen**

```python
# In examples/real_data_demo.py, model config
config = {
    'd_model': 512,
    'n_heads': 8,
    'vocab_size': 10000,
    'dropout': 0.3,        # ERHÖHEN von 0.1 auf 0.3!
    'k_permutations': 20,
    'epsilon': 0.05,
}

# Weight Decay im Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01  # NEU: L2 Regularization
)
```

**Effekt**: Verhindert Overfitting, zwingt Modell zu generalisieren

---

#### 3. **Early Stopping implementieren**

```python
# Nach Zeile ~465 in real_data_demo.py
patience = 3
best_val_acc = 0.0
patience_counter = 0

for epoch in range(n_epochs):
    # ... training ...

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save checkpoint
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
```

**Effekt**: Stoppt Training bevor Overfitting zu stark wird

---

#### 4. **Learning Rate Scheduler**

```python
# Nach Optimizer-Definition
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2,
    verbose=True
)

# Im Training Loop nach Validation
scheduler.step(val_acc)
```

**Effekt**: Reduziert Learning Rate wenn kein Fortschritt → bessere Konvergenz

---

### 🔬 Mittelfristig (2-3 Stunden)

#### 5. **Data Augmentation für Text**

```python
import random

def augment_text(text, p=0.1):
    """Randomly drop words to create variation."""
    words = text.split()
    augmented = [w for w in words if random.random() > p]
    return ' '.join(augmented)

# In DataLoader collate_fn
texts = [augment_text(item['text']) for item in batch]
```

**Effekt**: Erhöht effektive Trainingsdaten-Größe

---

#### 6. **Kleineres Modell testen**

```python
# Reduziere Modell-Größe für kleine Datasets
config = {
    'd_model': 256,      # von 512
    'n_heads': 4,        # von 8
    'vocab_size': 5000,  # von 10000
    'dropout': 0.3,
    'k_permutations': 10,  # von 20
}
```

**Parameter-Reduktion**: 4.2M → ~1.2M Parameter

**Ratio**: 1,200,000 / 20,000 samples = 60 Parameter/Sample ✅ (viel besser!)

---

#### 7. **Cross-Validation**

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold+1}/5")
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Train model on train_subset
    # Validate on val_subset
    # Average results across folds
```

**Effekt**: Robustere Performance-Schätzung

---

### 🚀 Langfristig (1-2 Wochen)

#### 8. **Transfer Learning**

```python
from transformers import AutoModel

# Nutze pre-trained BERT als Encoder
bert = AutoModel.from_pretrained('bert-base-uncased')

# Freeze BERT, train nur Bayesian Layer
for param in bert.parameters():
    param.requires_grad = False
```

**Effekt**: Nutzt pre-trained Knowledge → weniger Daten nötig

---

#### 9. **Uncertainty Calibration**

```python
from sklearn.calibration import calibration_curve

# Nach Training
y_true, y_pred, uncertainties = [], [], []
for batch in test_loader:
    outputs = model(batch, return_uncertainty=True)
    y_pred.extend(outputs['predictions'])
    uncertainties.extend(outputs['epistemic_uncertainty'])

# Plot calibration
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true, y_pred, n_bins=10
)
```

**Effekt**: Zeigt ob Uncertainties aussagekräftig sind

---

#### 10. **Active Learning Loop**

```python
# Iterativ: Train → Find uncertain samples → Label → Retrain
for iteration in range(10):
    # Train current model
    model.train()

    # Find most uncertain unlabeled samples
    unlabeled_dataset = load_dataset('imdb', split='unsupervised')
    uncertainties = predict_uncertainties(model, unlabeled_dataset)
    top_uncertain = np.argsort(uncertainties)[-100:]

    # Simulate human labeling (in production: actual humans)
    new_labels = get_human_labels(unlabeled_dataset[top_uncertain])

    # Add to training set
    train_dataset = train_dataset + new_labels
```

**Effekt**: Maximiert Lernen pro gelabeltem Sample

---

## 📊 Erwartete Verbesserungen

### Nach Quick Wins (1 Stunde):

| Metrik | Vorher | Nachher | Status |
|--------|--------|---------|--------|
| Train Acc | 100% ❌ | 85-92% ✅ | Realistisch |
| Val Acc | 100% ❌ | 80-88% ✅ | Generalisiert |
| Loss | 0.0000 ❌ | 0.15-0.35 ✅ | Gesund |
| Uncertainty Range | 0.0556±0.0001 ❌ | 0.05-0.30 ✅ | Variabel |
| Overfitting | Extrem ❌ | Minimal ✅ | OK |

### Nach Mittelfristig (3 Stunden):

| Metrik | Ziel |
|--------|------|
| Val Acc | 88-93% |
| F1 Score | 0.87-0.91 |
| ECE (Calibration) | <0.1 |
| Uncertain Samples Identified | 10-20% |

---

## 🎯 Prioritäten-Roadmap

### 🔴 Kritisch (JETZT):
1. **Mehr Daten**: 1000 → 20,000 Samples
2. **Dropout erhöhen**: 0.1 → 0.3
3. **Weight Decay**: 0.01

### 🟡 Wichtig (Diese Woche):
4. Early Stopping
5. Learning Rate Scheduler
6. Kleineres Modell (für 1K samples)

### 🟢 Nice-to-Have (Nächste Woche):
7. Data Augmentation
8. Cross-Validation
9. Transfer Learning

---

## 📝 Code-Änderungen (Copy-Paste Ready)

### Datei: `examples/real_data_demo.py`

```python
# Zeile ~400: MEHR DATEN
max_samples_train = 20000  # statt 1000
max_samples_test = 5000    # statt 200

# Zeile ~410: MEHR DROPOUT
config = {
    'd_model': 512,
    'n_heads': 8,
    'vocab_size': 10000,
    'dropout': 0.3,  # statt 0.1
    'k_permutations': 20,
    'epsilon': 0.05,
}

# Zeile ~416: WEIGHT DECAY
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01  # NEU
)

# NEU: Learning Rate Scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True
)

# Im Training Loop nach Validation (Zeile ~465):
scheduler.step(val_acc)

# NEU: Early Stopping
patience = 3
patience_counter = 0

if val_acc > best_val_acc:
    best_val_acc = val_acc
    patience_counter = 0
else:
    patience_counter += 1

if patience_counter >= patience:
    print(f"Early stopping at epoch {epoch+1}")
    break
```

---

## ✅ Success Criteria (nach Fixes)

### Training sollte zeigen:
- [ ] Training Accuracy: 85-92% (nicht 100%!)
- [ ] Validation Accuracy: 80-88%
- [ ] Loss: 0.15-0.35 (nicht 0.0000!)
- [ ] Uncertainty Range: 0.05-0.30 (variabel!)
- [ ] 10-20% Samples als "uncertain" identifiziert
- [ ] ECE (Calibration Error) < 0.1

---

## 🔍 Debugging: Ist Overfitting behoben?

### Quick Check nach Training:

```python
# Gap zwischen Train und Val Accuracy sollte <5% sein
gap = train_acc - val_acc
if gap < 0.05:
    print("✅ Kein Overfitting!")
elif gap < 0.10:
    print("⚠️ Leichtes Overfitting")
else:
    print("❌ Starkes Overfitting - mehr Regularization!")

# Uncertainty Varianz sollte >0.01 sein
uncertainty_std = np.std(uncertainties)
if uncertainty_std > 0.01:
    print("✅ Uncertainty funktioniert!")
else:
    print("❌ Uncertainty zu homogen - Model zu confident!")
```

---

**Nach Implementierung dieser Fixes wird dein Modell realistisch trainieren und echte Uncertainty Quantification liefern!** 🎯
