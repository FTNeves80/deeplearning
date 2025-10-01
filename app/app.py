# app.py — UI de pedidos + recomendação usando modelo TensorFlow já treinado
import json
from pathlib import Path
import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
PATH_MODEL_DIR = ROOT / "artifacts" / "model"
PATH_MAP_DIR = ROOT / "artifacts" / "mappings"
PATH_CATALOG = [
    ROOT / "data" / "catalog" / "products.csv",
    Path.cwd().parents[0] / "data" / "catalog" / "products.csv",
    Path("products.csv"),
]

# -------------------- carregamento de recursos --------------------
def load_catalog() -> pd.DataFrame:
    for p in PATH_CATALOG:
        if p.exists():
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            pid_col  = cols.get("product_id") or cols.get("codigo_produto")
            name_col = cols.get("name")       or cols.get("descricao_produto")
            cat_col  = cols.get("category")   or cols.get("categoria")
            price_col= cols.get("price")      or cols.get("preco")
            if not (pid_col and name_col and cat_col and price_col):
                raise ValueError("CSV não tem as colunas esperadas (product_id, name, category, price)")
            df = df.rename(columns={
                pid_col: "product_id",
                name_col: "name",
                cat_col: "category",
                price_col: "price",
            })
            df["product_id"] = df["product_id"].astype(int)
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
            return df[["product_id", "name", "category", "price"]]

    raise FileNotFoundError("Nenhum catálogo encontrado em PATH_CATALOG")

def load_mappings():
    with open(PATH_MAP_DIR / "product_to_int.json", "r", encoding="utf-8") as f:
        product_to_int = json.load(f)
    with open(PATH_MAP_DIR / "int_to_product.json", "r", encoding="utf-8") as f:
        int_to_product = json.load(f)  # chaves string → ok
    with open(PATH_MAP_DIR / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    max_len = int(meta.get("max_len", 10))
    return product_to_int, int_to_product, max_len

# cacheia para não recarregar a cada clique

def load_model():
    # prioriza .keras; fallback .h5
    candidates = [PATH_MODEL_DIR / "melhor_modelo.keras", PATH_MODEL_DIR / "melhor_modelo.h5"]
    for c in candidates:
        if c.exists():
            return tf.keras.models.load_model(str(c))
    raise FileNotFoundError("Modelo não encontrado em artifacts/model/ (melhor_modelo.keras ou .h5).")

CATALOG = load_catalog()
NAME_TO_PID = dict(zip(CATALOG["name"], CATALOG["product_id"]))
PID_TO_NAME = dict(zip(CATALOG["product_id"], CATALOG["name"]))
MODEL = load_model()
PRODUCT_TO_INT, INT_TO_PRODUCT, MAX_LEN = load_mappings()

# -------------------- helpers de carrinho e predição --------------------
def make_editor_df() -> pd.DataFrame:
    df = CATALOG.copy()
    df["quantity"] = 0
    return df.rename(columns={"name": "Produto", "category": "Categoria", "price": "Preço", "quantity": "Quantidade"})

def summarize(df_editor):
    df = pd.DataFrame(df_editor, columns=["Produto", "Categoria", "Preço", "Quantidade"])
    df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0).astype(int)
    df["Preço"] = pd.to_numeric(df["Preço"], errors="coerce").fillna(0.0)
    cart = df[df["Quantidade"] > 0].copy()
    subtotal = float((cart["Preço"] * cart["Quantidade"]).sum())
    n_items = int(cart["Quantidade"].sum())
    return df, cart, n_items, subtotal

def build_sequence_ids(cart: pd.DataFrame, max_len: int) -> np.ndarray:
    seq = []
    for _, row in cart.iterrows():
        name = str(row["Produto"])
        q = int(row["Quantidade"])
        codigo = NAME_TO_PID.get(name)                  # codigo_produto
        pid_int = PRODUCT_TO_INT.get(str(codigo)) or PRODUCT_TO_INT.get(codigo)
        if pid_int is None:
            continue
        seq.extend([pid_int] * max(0, q))

    if not seq:
        return np.zeros((1, max_len), dtype=np.float32)
    seq = seq[-max_len:]
    if len(seq) < max_len:
        seq = [0] * (max_len - len(seq)) + seq
    return np.array(seq, dtype=np.float32).reshape(1, max_len)

def predict_topk(df_editor, topk):
    df, cart, n_items, subtotal = summarize(df_editor)

    # monta sequência para o modelo
    x = build_sequence_ids(cart, MAX_LEN)

    # chama o modelo
    scores = MODEL.predict(x, verbose=0)  # shape: (1, vocab)
    scores = scores.squeeze().astype(float)

    # mascarar produtos já no carrinho (não sugerir o que já foi escolhido)
    already = set(cart["Produto"].tolist())
    mask = np.ones_like(scores, dtype=float)

    # ids presentes no carrinho
    for name in already:
        codigo = NAME_TO_PID.get(name)
        pid_int = PRODUCT_TO_INT.get(str(codigo)) or PRODUCT_TO_INT.get(codigo)
        if pid_int is not None and 0 <= pid_int < len(scores):
            mask[pid_int] = 0.0


    mask[0] = 0.0              # PAD fora
    scores = scores * mask
    s = scores.sum()
    if s > 0:
        scores = scores / s    # re-normaliza (igual ao notebook)

    # ordenar desc e pegar top-k válidos
    idx_sorted = np.argsort(scores)[::-1]
    out = []
    k = int(topk)
    for idx in idx_sorted:
        if scores[idx] <= 0:
            continue
        # converter id → nome
        codigo = INT_TO_PRODUCT.get(str(idx))
        if codigo is None: 
            continue
        nome = PID_TO_NAME.get(int(codigo))
        if not nome:
            continue
        out.append([nome, float(scores[idx])])
        if len(out) >= k:          # <- pare no Top-K
            break

    if not out:
        out = [["—", 0.0]]

    summary_md = (
        f"**Itens no carrinho:** {n_items}  \n"
        f"**Subtotal:** R$ {subtotal:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )
    return out, summary_md

def on_clear(df_editor):
    df = pd.DataFrame(df_editor, columns=["Produto", "Categoria", "Preço", "Quantidade"])
    df["Quantidade"] = 0
    return df

# -------------------- UI --------------------
with gr.Blocks(title="Pedidos • Recomendador") as demo:
    gr.Markdown("## Monte seu pedido\nDefina as **quantidades** por produto e clique em **Sugerir** para ver recomendações do modelo.")

    editor = gr.Dataframe(
        value=make_editor_df(),
        headers=["Produto", "Categoria", "Preço", "Quantidade"],
        datatype=["str", "str", "number", "number"],
        row_count=(len(CATALOG), "fixed"),
        col_count=(4, "fixed"),
        interactive=True
    )

    with gr.Row():
        topk = gr.Slider(1, 10, value=3, step=1, label="Top-K sugestões")
        btn_suggest = gr.Button("Sugerir", variant="primary")
        btn_clear = gr.Button("Limpar quantidades")

    recs = gr.Dataframe(headers=["Produto", "Score"], datatype=["str", "number"], label="Sugestões")
    summary = gr.Markdown("**Itens no carrinho:** 0  \n**Subtotal:** R$ 0,00")

    btn_suggest.click(fn=predict_topk, inputs=[editor, topk], outputs=[recs, summary])
    btn_clear.click(fn=on_clear, inputs=[editor], outputs=[editor])

# 🔽 Aqui entra o quadro que você pediu

    gr.Markdown("---")
    gr.Markdown("### Trabalho de DeepLearning")
    gr.Markdown(
            """
            - Bruno Bersan  
            - Ingrid Coda
            - Leonardo Cunha
            - Felipe Neves
            - Cris Andrade
            """
        )

if __name__ == "__main__":
    demo.launch()
