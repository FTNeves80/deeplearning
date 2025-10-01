# 🛒 Sistema de Recomendação de Produtos com Deep Learning

Trabalho desenvolvido em grupo para a disciplina de **Deep Learning**.  
O objetivo é criar uma aplicação web simples onde o usuário monta um carrinho de compras e o modelo sugere os próximos produtos mais prováveis de serem adicionados, com base em dados históricos.

---

## 📂 Estrutura do Projeto

```bash
reco_app/
├─ app/                # Código da interface (Gradio)
│   └─ app.py
├─ artifacts/          # Artefatos do modelo
│   ├─ model/          # Modelo salvo (.keras)
│   └─ mappings/       # JSONs de mapeamento + meta
├─ data/
│   ├─ catalog/        # Catálogo de produtos (CSV)
│   └─ source/         # Dados brutos (ex.: Pedidos.xlsx)
└─ env/                # Ambiente Conda (local)

⚙️ 1. Preparação do Ambiente

No CMD do Windows:

# Criar pasta do projeto
mkdir reco_app
cd reco_app

Este repositório inclui um `environment.yml` para recriar o ambiente idêntico ao usado no projeto.  
> Observação: **não** suba a pasta `env/` para o Git (já há `.gitignore`). Versione apenas o `environment.yml`.

### Criar o ambiente (na raiz do projeto)
```bash
# cria o ambiente localmente dentro de ./env
conda env create -f environment.yml -p ./env
conda activate ./env

# Ativar ambiente
conda init cmd.exe
conda activate E:\reco_app\env

# Instalar dependências principais
pip install tensorflow==2.17.*
pip install gradio
pip install pandas numpy openpyxl




💡 Dica: exporte as libs com pip freeze > requirements.txt para reprodutibilidade.

📁 2. Organização dos Artefatos

Modelo salvo:
artifacts/model/modelo_total_dados_ULTIMO.keras

Mapeamentos (JSONs):

artifacts/mappings/product_to_int.json → codigo_produto → id interno

artifacts/mappings/int_to_product.json → id interno → codigo_produto

artifacts/mappings/meta.json → parâmetros do modelo (ex.: {"max_len": 10})

Catálogo de produtos (CSV):
data/catalog/products.csv com colunas:

product_id,name,category,price
712,DOGAO 1,Lanche,15.0
752,CONTI COLA 2 LITROS,Bebida,10.0
743,SACHE DE MAIONESE,Acompanhamento,2.0

🧪 3. Teste Rápido do Modelo

Para confirmar que o modelo carrega:

cd app
python model_smoke_test.py


Saída esperada:

✅ [OK] Modelo carregado

✅ Entrada (None, 10) → significa max_len=10

✅ Saída (batch, N) → N = nº de produtos possíveis

🌐 4. Executando a Aplicação

Arquivo principal: app/app.py

cd app
python app.py


Acesse em: 👉 http://127.0.0.1:7860

Funcionalidades:

Selecionar quantidades de cada produto no carrinho

Calcular subtotal automaticamente

Gerar sugestões Top-K com base no modelo

Mostrar produtos recomendados + scores

⚠️ 5. Pontos de Atenção

Os JSONs devem ser exportados do mesmo notebook de treino.

O products.csv deve usar os mesmos codigo_produto que aparecem nos JSONs.

O ID 0 é reservado para padding → nunca será sugerido.

Para trocar de modelo, basta substituir o arquivo em artifacts/model/.