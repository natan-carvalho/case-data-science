import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocessing import clean_text

# 1. Carregar os dados
df = pd.read_csv('data/articles.csv')

# 2. Remover linhas com valores nulos na coluna 'text'
df = df.dropna(subset=['text'])

# 3. Converter a coluna 'date' para datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 4. Removendo categorias com poucas amostras
frequencia_categorias = df['category'].value_counts(normalize=True)
valida_categorias = frequencia_categorias[frequencia_categorias >= 0.001].index
df = df[df['category'].isin(valida_categorias)]

# 5. Limpar os textos
df['title'] = df['title'].astype(str).apply(clean_text)
df['category'] = df['category'].astype(str).apply(clean_text)

# 6. Separar variáveis
X = df['title']
y = df['category']

# 7. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Vetorização
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 9. Treinar o modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 10. Avaliar o modelo
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 11. Criar a pasta models se não existir e salvar o modelo e o vetorizer
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')