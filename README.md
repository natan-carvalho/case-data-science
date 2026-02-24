# Case Data Science - Classificação de Notícias

Este projeto consiste em uma API desenvolvida com **FastAPI** para classificar títulos de notícias em categorias pré-definidas. O sistema utiliza um modelo de Machine Learning treinado com **Regressão Logística** sobre dados textuais processados.

## 📋 Estrutura do Projeto

```
case-data-science/
├── api/
│   └── main.py          # Aplicação FastAPI e endpoints
├── data/
│   └── articles.csv     # Dataset base (necessário para o treino)
├── notebooks/
│   └── eda.ipynb        # Análise Exploratória dos Dados
├── models/
│   ├── model.pkl        # Modelo treinado (serializado)
│   └── vectorizer.pkl   # Vetorizador TF-IDF (serializado)
├── src/
│   ├── preprocessing.py # Lógica de limpeza e normalização de texto
│   ├── train.py         # Pipeline de treinamento e avaliação do modelo
│   └── predict.py       # Script auxiliar para inferência
├── Dockerfile           # Configuração da imagem Docker
└── requirements.txt     # Dependências do projeto
```

## 🚀 Como Executar

### Pré-requisitos
- **Docker** (Recomendado)
- Ou **Python 3.12+** configurado localmente.

### Opção 1: Via Docker (Recomendado)

O `Dockerfile` deste projeto foi configurado para instalar as dependências e **executar o treinamento do modelo automaticamente** durante a construção da imagem.

1. **Construir a imagem:**
   ```bash
   docker build -t case-data-science .
   ```

2. **Rodar o container:**
   ```bash
   docker run -p 8000:8000 case-data-science
   ```

A API estará disponível em `http://localhost:8000`.

### Opção 2: Execução Local

1. **Adicionar a base de dados:**
   Certifique-se de que o arquivo `data/articles.csv` esteja presente no diretório `data/`. Este arquivo é essencial para o treinamento dos modelos.

2. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Treinar o modelo:**
   Você pode escolher entre duas versões de treinamento:
   - **Versão 1 (Regressão Logística):**
     ```bash
     python src/train.py
     ```
   - **Versão 2 (LinearSVC - SVM):**
     ```bash
     python src/train_v2.py
     ```

4. **Iniciar a API:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
---
### Opção 3: Via Web (Cloud)

A API está publicada e pode ser testada diretamente pelo navegador através da documentação interativa (Swagger UI):

- **Link:** [https://news-category-api.onrender.com/docs](https://news-category-api.onrender.com/docs)

---
## 📡 Utilizando a API

A API oferece dois endpoints para predição, permitindo comparar os resultados dos modelos treinados:

### 1. Endpoint: `/predicao` (Logistic Regression)
Utiliza o modelo treinado via `src/train.py`.

**Exemplo com cURL:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predicao' \
  -H 'Content-Type: application/json' \
  -d '{ "title": "Novo avanço na tecnologia de inteligência artificial" }'
```

### 2. Endpoint: `/predicao_v2` (LinearSVC)
Utiliza o modelo treinado via `src/train_v2.py`, que geralmente apresenta melhor performance em textos.

**Exemplo com cURL:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predicao_v2' \
  -H 'Content-Type: application/json' \
  -d '{ "title": "Novo avanço na tecnologia de inteligência artificial" }'
```

## 🧠 Estratégia de Data Science

A etapa de **Análise Exploratória de Dados (EDA)** foi realizada no notebook `notebooks/eda.ipynb`. Nela, foram investigadas a distribuição das categorias, a presença de dados nulos e as características do texto, o que fundamentou as decisões de pré-processamento e modelagem.

### 1. Pré-processamento (`src/preprocessing.py`)
- **Limpeza:** O texto é convertido para minúsculas, números são removidos e a pontuação é eliminada.
- **Filtro de Dados:** Categorias com representatividade inferior a 0.1% no dataset são removidas para evitar desbalanceamento excessivo.
- **Split:** Divisão estratificada (80% treino / 20% teste).

### 2. Engenharia de Features
- **TF-IDF:** Utilizado para vetorização do texto com n-grams (1, 2).
- **V1 (`src/train.py`):** 5000 features.
- **V2 (`src/train_v2.py`):** 10000 features e `sublinear_tf=True` para melhor normalização.

### 3. Algoritmos
- **Modelo V1:** `LogisticRegression` - Baseline robusto e interpretável.
- **Modelo V2:** `LinearSVC` (SVM) - Geralmente superior para classificação de texto em alta dimensionalidade.

## ⛏️ Ferramentas Utilizadas

- Python - Linguagem de Programação
- FastAPI - Framework Web
- Scikit-Learn - Machine Learning
- Pandas - Manipulação de Dados
- Docker - Containerização

## ✍️ Autores <a name = "authors"></a>

- @natan-carvalho - Ideia & Trabalho Inicial