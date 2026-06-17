import pdfplumber
import unicodedata
import re
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words("spanish"))

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto).strip()
    return " ".join([p for p in texto.split() if p not in stop_words])

# 1. Ver todos los chunks
with pdfplumber.open("gokulab_info.pdf") as pdf:
    texto = ""
    for page in pdf.pages:
        contenido = page.extract_text()
        if contenido:
            texto += contenido + "\n"

chunks = [p.strip() for p in texto.split("\n") if len(p.strip()) >= 40]

print(f"Total chunks: {len(chunks)}\n")
for i, c in enumerate(chunks):
    print(f"[{i}] {c}")