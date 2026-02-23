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

1. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Treinar o modelo:**
   Execute o script de treino para gerar os arquivos `.pkl` na pasta `models/`. Certifique-se de que o arquivo `data/articles.csv` existe.
   ```bash
   python src/train.py
   ```

3. **Iniciar a API:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📡 Utilizando a API

### Endpoint: `/predicao` (POST)
Recebe um título de notícia e retorna a categoria prevista.

**Exemplo de Request (JSON):**
```json
{
  "title": "Mercado financeiro reage positivamente às novas taxas de juros"
}
```

**Exemplo com cURL:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predicao' \
  -H 'Content-Type: application/json' \
  -d '{ "title": "Novo avanço na tecnologia de inteligência artificial" }'
```

## 🧠 Estratégia de Data Science

A etapa de **Análise Exploratória de Dados (EDA)** foi realizada no notebook `notebooks/eda.ipynb`. Nela, foram investigadas a distribuição das categorias, a presença de dados nulos e as características do texto, o que fundamentou as decisões de pré-processamento e modelagem.

### 1. Pré-processamento (`src/preprocessing.py`)
- **Limpeza:** O texto é convertido para minúsculas, números são removidos e a pontuação é eliminada.
- **Filtro de Dados:** Categorias com representatividade inferior a 1% no dataset são removidas para evitar desbalanceamento excessivo.
- **Split:** Divisão estratificada (80% treino / 20% teste).

### 2. Engenharia de Features (`src/train.py`)
- **TF-IDF:** Utilizado para vetorização do texto.
  - **N-grams:** (1, 2) (considera palavras isoladas e pares de palavras).
  - **Max Features:** 5000 (limita o vocabulário às palavras mais relevantes).

### 3. Algoritmo
- **Modelo:** `LogisticRegression` (Scikit-Learn).
- **Motivo:** Escolhido por ser um baseline robusto, eficiente computacionalmente e oferecer boa interpretabilidade para classificação de texto.

## ⛏️ Ferramentas Utilizadas

- Python - Linguagem de Programação
- FastAPI - Framework Web
- Scikit-Learn - Machine Learning
- Pandas - Manipulação de Dados
- Docker - Containerização

## ✍️ Autores <a name = "authors"></a>

- @natan-carvalho - Ideia & Trabalho Inicial