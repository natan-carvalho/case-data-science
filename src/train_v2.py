import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from preprocessing import clean_text
import os

# Garantir que a pasta models existe
os.makedirs('models', exist_ok=True)

print("Iniciando treinamento v2 (SVM)...")

# 1. Carregar os dados
df = pd.read_csv('data/articles.csv')
df = df.dropna(subset=['text', 'title', 'category'])

# 2. Filtragem de categorias representativas (> 1%)
frequencia_categorias = df['category'].value_counts(normalize=True)
valida_categorias = frequencia_categorias[frequencia_categorias >= 0.001].index
df = df[df['category'].isin(valida_categorias)]

# 3. Limpar os textos (usando o preprocessing.py atualizado)
print("Limpando títulos...")
df['title_clean'] = df['title'].astype(str).apply(clean_text)

# 4. Separar variáveis
X = df['title_clean']
y = df['category']

# 5. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Vetorização Aprimorada
# Aumentamos o max_features e usamos unigramas e bigramas
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)

print("Vetorizando dados...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Treinar o modelo SVM (LinearSVC)
print("Treinando LinearSVC...")
model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
model.fit(X_train_vec, y_train)

# 8. Avaliar o modelo
y_pred = model.predict(X_test_vec)
print("Relatório de Classificação (SVM):")
print(classification_report(y_test, y_pred))

# 9. Salvar o novo modelo
joblib.dump(model, 'models/model_v2_svm.pkl')
joblib.dump(vectorizer, 'models/vectorizer_v2.pkl')

print("Modelo v2 salvo com sucesso em 'models/model_v2_svm.pkl'")
