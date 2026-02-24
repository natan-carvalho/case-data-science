import re
import string

# Lista básica de stopwords em português para melhoria do modelo
STOPWORDS = set([
    "a", "ao", "aos", "as", "através", "com", "como", "da", "das", "de", "do", "dos", "e", "em", "entre", 
    "era", "eram", "essa", "essas", "esse", "esses", "esta", "estas", "este", "estes", "foi", "fomos", 
    "foram", "fosse", "fossem", "há", "isso", "isto", "já", "mas", "meu", "meus", "minha", "minhas", 
    "na", "nas", "no", "nos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", 
    "por", "qual", "quando", "que", "se", "seu", "seus", "sua", "suas", "também", "teu", "teus", "tua", 
    "tuas", "um", "uma", "umas", "uns", "você", "vocês"
])

def clean_text(text: str) -> str:
    """
    Função aprimorada para limpeza de texto:
    - Conversão para minúsculas
    - Remoção de pontuação e números
    - Remoção de stopwords básicas
    - Normalização de espaços
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove números
    text = re.sub(r'\d+', '', text)
    # Remove pontuação
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenização simples e remoção de stopwords
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    
    return " ".join(words)
