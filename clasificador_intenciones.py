import os
import re
import pickle
import unicodedata
import string
import gdown
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

stop_words = set(stopwords.words("spanish"))
MODEL_PATH = "modelo_intents.pkl"


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto).strip()
    return " ".join([p for p in texto.split() if p not in stop_words])


def entrenar_y_guardar():
    file_id = "1mzmYKXunfzqSBT-Z6lZ1MSAYogljt0fm"
    file_name = "nuevo_dataset.xlsx"
    url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"

    if not os.path.exists(file_name):
        print("Descargando dataset...")
        gdown.download(url, file_name, quiet=False)

    df = pd.read_excel(file_name, engine="openpyxl")
    df = df.drop(columns=["Marca temporal", "Dirección de correo electrónico"], errors="ignore")

    df_final = pd.melt(df, value_vars=df.columns, var_name="Intent", value_name="Texto")
    df_final["Intent"] = df_final["Intent"].str.strip().replace({
        "Escribe cómo preguntarías la dirección o ubicación": "Consultar_Ubicacion",
        "Escribe cómo le preguntarías a la academia cuánto cuestan los cursos": "Consultar_Costos",
        "Escribe cómo preguntarías qué horarios manejan": "Consultar_Horarios",
        "Escribe cómo preguntarías si otorgan algún certificado o diploma": "Consultar_Certificacion",
        "Escribe un saludo inicial": "Saludo",
        "Escribe cómo escribirías una despedida": "Despedida",
        "Escribe cómo preguntarías si las clases son virtuales, presenciales o mixtas": "Consultar_Modalidad",
        "Escribe cómo pedirías información sobre qué cursos tienen disponibles": "Consultar_Cursos",
        "Escribe cómo pedirías una clase demo o de prueba antes de inscribirte": "Consultar_ClaseDemo",
        "Escribe cómo preguntarías si hay edad mínima para tomar el curso": "Consultar_RequisitosEdad",
        "Escribe cómo preguntarías las formas de pago": "Consultar_FormasPago",
        "Escribe cómo preguntarías la duración de los cursos": "Consultar_Duracion",
    })
    df_final = df_final.dropna(subset=["Texto"])
    df_final["Texto"] = df_final["Texto"].apply(limpiar_texto)

    vec = TfidfVectorizer()
    X = vec.fit_transform(df_final["Texto"])
    Y = df_final["Intent"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    gs = GridSearchCV(
        SVC(probability=True),
        {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    gs.fit(X, Y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"modelo": gs.best_estimator_, "vectorizer": vec}, f)

    print(f"Modelo entrenado. Mejor config: {gs.best_params_}")
    return gs.best_estimator_, vec, df_final


def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        print("Modelo cargado desde disco.")
        with open(MODEL_PATH, "rb") as f:
            datos = pickle.load(f)
        return datos["modelo"], datos["vectorizer"]
    print("No se encontró modelo en disco. Entrenando...")
    modelo, vec, _ = entrenar_y_guardar()
    return modelo, vec


mejor_modelo, vectorizer = cargar_modelo()


def predecir_intent(texto, umbral=0.5, umbral_secundario=0.35):
    if mejor_modelo is None or vectorizer is None:
        return ["Desconocido"], [0.0]

    vector   = vectorizer.transform([limpiar_texto(texto)])
    probs    = mejor_modelo.predict_proba(vector)[0]
    clases   = mejor_modelo.classes_
    max_prob = max(probs)

    if max_prob < umbral:
        return ["Desconocido"], [max_prob]

    pares = sorted(zip(clases, probs), key=lambda x: -x[1])
    intencion_principal = pares[0][0]

    if intencion_principal in ["Saludo", "Despedida"]:
        return [intencion_principal], [pares[0][1]]

    intenciones = []
    confianzas  = []
    for clase, prob in pares:
        if clase in ["Saludo", "Despedida"]:
            continue
        if prob >= umbral_secundario:
            intenciones.append(clase)
            confianzas.append(prob)
        if len(intenciones) == 3:
            break

    if not intenciones:
        return ["Desconocido"], [max_prob]

    return intenciones, confianzas


if __name__ == "__main__":
    # ── Recargar datos para métricas ─────────────────────────────────────────
    file_name = "nuevo_dataset.xlsx"
    df = pd.read_excel(file_name, engine="openpyxl")
    df = df.drop(columns=["Marca temporal", "Dirección de correo electrónico"], errors="ignore")

    df_final = pd.melt(df, value_vars=df.columns, var_name="Intent", value_name="Texto")
    df_final["Intent"] = df_final["Intent"].str.strip().replace({
        "Escribe cómo preguntarías la dirección o ubicación": "Consultar_Ubicacion",
        "Escribe cómo le preguntarías a la academia cuánto cuestan los cursos": "Consultar_Costos",
        "Escribe cómo preguntarías qué horarios manejan": "Consultar_Horarios",
        "Escribe cómo preguntarías si otorgan algún certificado o diploma": "Consultar_Certificacion",
        "Escribe un saludo inicial": "Saludo",
        "Escribe cómo escribirías una despedida": "Despedida",
        "Escribe cómo preguntarías si las clases son virtuales, presenciales o mixtas": "Consultar_Modalidad",
        "Escribe cómo pedirías información sobre qué cursos tienen disponibles": "Consultar_Cursos",
        "Escribe cómo pedirías una clase demo o de prueba antes de inscribirte": "Consultar_ClaseDemo",
        "Escribe cómo preguntarías si hay edad mínima para tomar el curso": "Consultar_RequisitosEdad",
        "Escribe cómo preguntarías las formas de pago": "Consultar_FormasPago",
        "Escribe cómo preguntarías la duración de los cursos": "Consultar_Duracion",
    })
    df_final = df_final.dropna(subset=["Texto"])
    df_final["Texto"] = df_final["Texto"].apply(limpiar_texto)

    X = vectorizer.transform(df_final["Texto"])
    Y = df_final["Intent"]

    # ── Métricas con cross-validation ────────────────────────────────────────
    print("\n===== MÉTRICAS CON CROSS-VALIDATION (5 folds) =====")
    y_pred_cv = cross_val_predict(mejor_modelo, X, Y, cv=5)
    print(classification_report(Y, y_pred_cv, zero_division=0))

    # ── Matriz de confusión ───────────────────────────────────────────────────
    print("\n===== MATRIZ DE CONFUSIÓN =====")
    cm = confusion_matrix(Y, y_pred_cv, labels=mejor_modelo.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mejor_modelo.classes_)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Matriz de confusión — Gōku Lab Chatbot")
    plt.tight_layout()
    plt.savefig("matriz_confusion.png", dpi=150)
    plt.show()
    print("Imagen guardada como matriz_confusion.png")

    # ── Ejemplos de prueba ────────────────────────────────────────────────────
    print("\n===== EJEMPLOS DE PRUEBA =====")
    ejemplos = [
        # Duración
        "cuánto dura cada clase?",
        "de cuánto tiempo son las sesiones?",
        "las clases son largas?",
        # Costos
        "cuánto cuesta el curso?",
        "cuál es el precio?",
        # Horarios
        "qué horarios manejan?",
        "a qué hora son las clases?",
        # Saludo
        "hola buenas tardes",
        "qué tal, buenos días",
        # Cursos
        "qué cursos tienen disponibles?",
        "me pueden decir qué cursos ofrecen?",
        # Modalidad
        "las clases son presenciales o en línea?",
        # ClaseDemo
        "puedo tomar una clase de prueba?",
        # Ubicacion
        "dónde están ubicados?",
        # Despedida
        "muchas gracias, hasta luego",
        # Casos difíciles
        "oye",
        "ok",
        "mm",
        "cuánto y dónde están?",
    ]

    for texto in ejemplos:
        intents, confs = predecir_intent(texto)
        confs_str = ", ".join([f"{c:.0%}" for c in confs])
        print(f"  '{texto}'")
        print(f"   → {' + '.join(intents)} ({confs_str})\n")

# prueba con mensajes multi-intención
ejemplos_multi = [
    "¿cuánto cuesta el curso y qué horarios tienen?",
    "hola, ¿dónde están ubicados y cuánto cuesta inscribirse?",
    "buenas, ¿qué cursos tienen y cuánto me saldría?",
    "¿tienen clases en línea y cuánto cuestan?",
    "oye, ¿cuánto dura cada clase y a qué hora son?",
    "me interesa saber los horarios y si dan certificado",
    "hola, ¿dónde quedan y qué cursos tienen?",
    "¿cuánto cuesta y dan certificado al terminar?",
    "buenas, ¿hay clase demo y cuánto cuesta inscribirse?",
    "¿a qué edad pueden entrar y cuánto sale el curso?",
    "¿las clases son presenciales y cuánto duran?",
    "oye, ¿qué cursos tienen y a qué hora son?",
    "¿cuánto cuesta, dónde están y qué cursos dan?",
    "hola, ¿dan certificado y cuáles son las formas de pago?",
    "¿puedo tomar una clase demo y cuánto cuesta después?",
]

for texto in ejemplos_multi:
    intents, confs = predecir_intent(texto)
    confs_str = ", ".join([f"{c:.0%}" for c in confs])
    print(f"  '{texto}'")
    print(f"   → {' + '.join(intents)} ({confs_str})\n")