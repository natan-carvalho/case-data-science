import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocessing import clean_text, remove_values_null

# 1. Carregar os dados
df = pd.read_csv('data/articles.csv')

# 2. Remover linhas com valores nulos
df = remove_values_null(df, 'text')

# 3. Limpar os textos
df['title'] = df['title'].astype(str).apply(clean_text)
df['category'] = df['category'].astype(str).apply(clean_text)

# 4. Separar variáveis
X = df['title']
y = df['category']

# 5. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Vetorização
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Treinar o modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 8. Avaliar o modelo
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 9. Salvar o modelo e o vetorizer
joblib.dump(model, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')