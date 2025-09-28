# 🛒 Sistema de Recomendação de Produtos com Deep Learning

Trabalho desenvolvido em grupo para disciplina de Deep Learning.  
O objetivo é criar uma aplicação web simples onde o usuário monta um carrinho de compras e o modelo sugere o próximo produto com maior probabilidade de ser adicionado, baseado em dados históricos.

---

## 🚀 Estrutura do Projeto

```bash
reco_app/
├─ app/                # Código da interface (Gradio)
│   └─ app.py
├─ artifacts/          # Artefatos do modelo
│   ├─ model/          # Modelo salvo (.keras ou .h5)
│   └─ mappings/       # JSONs de mapeamento + meta
├─ data/
│   ├─ catalog/        # Catálogo de produtos (CSV)
│   └─ source/         # Dados brutos (ex.: Pedidos.xlsx)
└─ env/                # Ambiente Conda (local)



---

## 🖥️ 1. Preparação do Ambiente

No **CMD do Windows**:

```bash
# Criar pasta do projeto
mkdir reco_app
cd reco_app

# Criar ambiente conda dentro da pasta
conda create -p ./env python=3.11 -y

# Ativar ambiente
conda init cmd.exe
conda activate E:\reco_app\env

pip install tensorflow==2.17.*
pip install gradio
pip install pandas numpy openpyxl

📂 3. Organização das Pastas e Arquivos

Modelo salvo:
Colocar em artifacts/model/melhor_modelo.keras (ou .h5).

Mapeamentos (em JSON):

artifacts/mappings/product_to_int.json → nome → id

artifacts/mappings/int_to_product.json → id → nome

artifacts/mappings/meta.json → parâmetros (ex.: {"max_len": 10})

Catálogo de produtos (CSV):

Em data/catalog/products.csv

Deve ter colunas: name,category,price

Exemplo mínimo de products.csv:

name,category,price
Cachorro-quente,Lanche,15.0
Coca-Cola,Bebida,7.0
Batata palha,Acompanhamento,6.0

4. Teste Rápido do Modelo

Para confirmar que o modelo carrega:

cd app
python model_smoke_test.py

Saída esperada:

Modelo carregado

Entrada (None, 10) → significa max_len=10

Saída (batch, 376) → 376 produtos possíveis

🌐 5. Executando a Aplicação

Arquivo principal: app/app.py

cd app
python app.py


Acesse em: http://127.0.0.1:7860

Funcionalidades:

Escolher quantidades de cada produto no carrinho

Calcular subtotal

Gerar sugestões com base no modelo (Top-K)

Mostrar nomes e scores ordenados

🛠️ 6. Pontos de Atenção

Os JSONs precisam ser exportados do mesmo notebook de treino para garantir que os IDs dos produtos correspondem ao que o modelo aprendeu.

Se IDs começarem em 1 (em vez de 0), ajustar no app.py (offset).

O products.csv deve usar os mesmos nomes que aparecem nos JSONs.

👥 Equipe

Trabalho de Deep Learning desenvolvido por:

Bruno

Ingrid

Leonardo

Felipe

Cris


👉 Agora sim, está tudo em **uma única caixa** para copiar e colar.  

Quer que eu adicione no final um **passo extra** mostrando como exportar os JSONs a partir do notebook de treino (para ninguém esquecer quando for reproduzir)?
