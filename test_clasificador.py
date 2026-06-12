import os
import re
import pickle
import unicodedata
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)


MODEL_PATH = "modelo_intents.pkl"
stop_words = set(stopwords.words("spanish"))


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto).strip()
    return " ".join([p for p in texto.split() if p not in stop_words])


def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print(f"no se encontró {MODEL_PATH}")
        print("asegúrate de correr este script desde la misma carpeta donde está el modelo.")
        exit(1)
    with open(MODEL_PATH, "rb") as f:
        datos = pickle.load(f)
    print(f"modelo cargado desde {MODEL_PATH}\n")
    return datos["modelo"], datos["vectorizer"]


modelo, vectorizer = cargar_modelo()

def probar_frase(texto, umbral=0.5, umbral_secundario=0.35):
    texto_limpio = limpiar_texto(texto)
    vector       = vectorizer.transform([texto_limpio])
    probs        = modelo.predict_proba(vector)[0]
    clases       = modelo.classes_

    # Ordenar por probabilidad descendente
    pares = sorted(zip(clases, probs), key=lambda x: -x[1])

    print(f"\n{'─'*55}")
    print(f"  Frase:   {texto}")
    print(f"  Limpia:  {texto_limpio}")
    print(f"{'─'*55}")
    print(f"  {'INTENCIÓN':<30} {'CONFIANZA':>10}  {'ESTADO'}")
    print(f"  {'─'*30} {'─'*10}  {'─'*15}")

    for clase, prob in pares[:5]:  # top 5
        if prob >= umbral:
            estado =  "DETECTADA"
        elif prob >= umbral_secundario:
            estado = "secundaria"
        else:
            estado = "   —"
        print(f"  {clase:<30} {prob:>9.1%}  {estado}")

    # Resultado final
    max_prob      = pares[0][1]
    intencion_top = pares[0][0]

    print(f"\n  → Resultado: ", end="")
    if max_prob >= umbral:
        print(f"{'CLASIFICADO como ' + intencion_top} ({max_prob:.1%})")
    else:
        print(f"DESCONOCIDO (confianza {max_prob:.1%} < umbral {umbral:.0%}) → usa RAG")


print("\n" + "="*55)
print("  TEST CLASIFICADOR — Gōku Lab")
print("\nFRASES DIRECTAS")
probar_frase("¿qué cursos tienen disponibles?")
probar_frase("¿cuánto cuestan los cursos?")
probar_frase("¿cuáles son los horarios?")
probar_frase("hola")
probar_frase("hasta luego")

print("\nFRASES CONVERSACIONALES")
probar_frase("Hola, buenas tardes. Tengo una pregunta, ¿cuánto cuestan los cursos?")
probar_frase("Me recomendaron la academia para mi hijo, ¿qué cursos tienen?")
probar_frase("Oye, quería saber si tienen clases de python y cuánto cuestan")
probar_frase("Buenas, me interesa inscribir a mi hijo, ¿cuál es la dirección?")
probar_frase("A ver, ¿tienen clases presenciales o solo en línea?")

print("\n📌 FRASES CON MÚLTIPLES INTENCIONES")
probar_frase("¿cuánto cuestan y cada cuándo son las clases?")
probar_frase("¿las clases son presenciales y dónde están ubicados?")
probar_frase("¿qué cursos tienen y hay clases para niños de 7 años?")

print("\n📌 FRASES AMBIGUAS")
probar_frase("¿tienen algo para niños?")
probar_frase("quiero más información")
probar_frase("está bien, gracias")

print("\n" + "="*55)
print("  MODO INTERACTIVO")
print("  Escribe una frase para probarla.")
print("  Escribe 'salir' para terminar.")
print("="*55)

while True:
    try:
        entrada = input("\n  Frase: ").strip()
        if entrada.lower() in ["salir", "exit", "quit", ""]:
            print("\n  Hasta luego")
            break
        probar_frase(entrada)
    except KeyboardInterrupt:
        print("\n\n  Hasta luego")
        break