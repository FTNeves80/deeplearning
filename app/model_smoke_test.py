# model_smoke_test.py
import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# 1) Local do modelo (prioriza .keras; fallback .h5)
ROOT = Path(__file__).resolve().parents[1]
M_DIR = ROOT / "artifacts" / "model"
CANDIDATES = [M_DIR / "melhor_modelo.keras", M_DIR / "melhor_modelo.h5"]

model_path = None
for c in CANDIDATES:
    if c.exists():
        model_path = c
        break

if not model_path:
    raise FileNotFoundError("Modelo não encontrado em artifacts/model/ (melhor_modelo.keras ou .h5).")

print(f"[INFO] Carregando modelo: {model_path}")
model = tf.keras.models.load_model(str(model_path))
print("[OK] Modelo carregado.")

# 2) Inspecionar assinatura de entrada
inputs = model.inputs
print(f"[INFO] Nº de entradas: {len(inputs)}")
for i, t in enumerate(inputs):
    print(f"  - input[{i}] name={t.name} shape={t.shape} dtype={t.dtype}")

# 3) Montar batch sintético (zeros) para testar predict
# Assumimos recomendador sequencial com entrada (batch, seq_len) de inteiros (IDs)
# Tentamos inferir seq_len da shape: (None, L)
seq_len = None
if len(inputs) == 1 and len(inputs[0].shape) == 2:
    seq_dim = inputs[0].shape[1]
    if seq_dim is not None:
        seq_len = int(seq_dim)

# Se não deu para inferir, tentamos valores comuns
candidate_seq_lens = [seq_len] if seq_len else [10, 20, 30]

ok = False
for L in candidate_seq_lens:
    if L is None:
        continue
    try:
        x = np.zeros((2, L), dtype=np.int32)  # dois exemplos, só zeros
        y = model.predict(x, verbose=0)
        print(f"[OK] Predict com seq_len={L}: output shape = {np.array(y).shape}")
        ok = True
        break
    except Exception as e:
        print(f"[WARN] Falhou com seq_len={L}: {e}")

if not ok:
    # Tentativa alternativa: modelos que esperam float
    for L in [10, 20, 30]:
        try:
            x = np.zeros((2, L), dtype=np.float32)
            y = model.predict(x, verbose=0)
            print(f"[OK] Predict (float) com seq_len={L}: output shape = {np.array(y).shape}")
            ok = True
            break
        except Exception as e:
            print(f"[WARN] Falhou (float) seq_len={L}: {e}")

if not ok:
    print("[ERRO] Não foi possível rodar predict com tentativas padrão.")
else:
    print("[SUCESSO] Smoke test concluído.")
