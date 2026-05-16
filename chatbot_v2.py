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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask_cors import CORS
import pdfplumber

# ─────────────────────────────────────────────────────────────
#  SETUP INICIAL
# ─────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
load_dotenv()

# ── Conexiones ────────────────────────────────────────────────
try:
    client_mongo = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
    client_mongo.server_info()
    db = client_mongo["chatbot_Goku_lab"]
    coleccion = db["conversaciones"]
    print("MongoDB conectado.")
except Exception as e:
    print(f"Error conectando a MongoDB: {e}")
    db = None
    coleccion = None

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
]
GROQ_KEYS = [k for k in GROQ_KEYS if k] 

if GROQ_KEYS:
    print(f"Groq conectado con {len(GROQ_KEYS)} key(s).")
else:
    print("No se encontraron API keys de Groq.")

# ── Analizador de sentimiento ─────────────────────────────────
analizador_sentimiento = SentimentIntensityAnalyzer()

# ─────────────────────────────────────────────────────────────
#  RAG — FALLBACK CON PDF (versión simple con pdfplumber)
# ─────────────────────────────────────────────────────────────
PDF_PATH = "gokulab_info.pdf"

def cargar_pdf():
    """Extrae todo el texto del PDF al arrancar."""
    if not os.path.exists(PDF_PATH):
        print(f"PDF no encontrado en: {PDF_PATH}")
        return ""
    try:
        texto = ""
        with pdfplumber.open(PDF_PATH) as pdf:
            for page in pdf.pages:
                texto += page.extract_text() + "\n"
        print(f"PDF cargado: {len(texto)} caracteres.")
        return texto
    except Exception as e:
        print(f"Error leyendo PDF: {e}")
        return ""

# Se carga una sola vez al arrancar, es instantáneo
CONTEXTO_PDF = cargar_pdf()

# ─────────────────────────────────────────────────────────────
#  MODELO: CARGAR O ENTRENAR
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "modelo_intents.pkl"
stop_words = set(stopwords.words("spanish"))


def limpiar_texto(texto):
    """Normaliza, limpia signos y quita stopwords."""
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto).strip()
    return " ".join([p for p in texto.split() if p not in stop_words])


def entrenar_y_guardar():
    """Descarga el dataset, entrena el modelo y lo guarda en disco."""
    file_id = "1viVnkIq_QIp8jI_Ysye6Q_WVerPsvUvE"
    file_name = "intencione.xlsx"
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
    return gs.best_estimator_, vec


def cargar_modelo():
    """Carga el modelo desde disco o entrena uno nuevo si no existe."""
    if os.path.exists(MODEL_PATH):
        print("Modelo cargado desde disco.")
        with open(MODEL_PATH, "rb") as f:
            datos = pickle.load(f)
        return datos["modelo"], datos["vectorizer"]
    print("⚙️  No se encontró modelo en disco. Entrenando...")
    return entrenar_y_guardar()


try:
    mejor_modelo, vectorizer = cargar_modelo()
except Exception as e:
    print(f"Error cargando/entrenando modelo: {e}")
    mejor_modelo, vectorizer = None, None


# ─────────────────────────────────────────────────────────────
#  UTILIDADES DE CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────

def predecir_intent(texto, umbral=0.5):
    """Clasifica el texto. Si confianza < umbral devuelve 'Desconocido'."""
    if mejor_modelo is None or vectorizer is None:
        return "Desconocido", 0.0
    vector = vectorizer.transform([limpiar_texto(texto)])
    probs = mejor_modelo.predict_proba(vector)[0]
    max_prob = max(probs)
    if max_prob < umbral:
        return "Desconocido", max_prob
    return mejor_modelo.classes_[probs.argmax()], max_prob


def obtener_datos_por_intencion(intencion):
    """Consulta MongoDB según la intención detectada."""
    if db is None:
        return {}

    config = db["datos_generales"].find_one({}, {"_id": 0}) or {}

    if intencion == "Consultar_Cursos":
        return {"cursos": list(db["cursos"].find({}, {"_id": 0})), "config": config}
    elif intencion == "Consultar_Costos":
        return {"cursos": list(db["cursos"].find({}, {"_id": 0, "nombreCurso": 1, "precio": 1, "moneda": 1})), "config": config}
    elif intencion == "Consultar_Horarios":
        return {"horarios": list(db["horarios"].find({}, {"_id": 0})), "config": config}
    elif intencion == "Consultar_Certificacion":
        return {"certificacion": config.get("certificacion"), "config": config}
    elif intencion == "Consultar_ClaseDemo":
        return {"masterclass": config.get("masterclass"), "config": config}
    elif intencion == "Consultar_FormasPago":
        return {"pagos": config.get("formas_pago"), "abonos": config.get("detalle_abonos"), "config": config}
    elif intencion == "Consultar_Modalidad":
        return {"cursos": list(db["cursos"].find({}, {"_id": 0, "nombreCurso": 1, "modalidad": 1})), "config": config}
    elif intencion == "Consultar_RequisitosEdad":
        return {"cursos": list(db["cursos"].find({}, {"_id": 0, "nombreCurso": 1, "edad_dirigida": 1})), "config": config}
    elif intencion == "Consultar_Duracion":
        return {"cursos": list(db["cursos"].find({}, {"_id": 0, "nombreCurso": 1, "duracion_min_por_clase": 1})), "config": config}
    elif intencion == "Consultar_Ubicacion":
        return {"direccion": config.get("direccion"), "referencias": config.get("referencias"), "link_maps": config.get("link_maps"), "config": config}

    return {"config": config}


# ─────────────────────────────────────────────────────────────
#  ANÁLISIS DE SENTIMIENTO
# ─────────────────────────────────────────────────────────────

def analizar_sentimiento(texto):
    """Usa VADER para detectar el tono emocional del mensaje."""
    scores = analizador_sentimiento.polarity_scores(texto)
    compound = scores["compound"]
    if compound <= -0.35:
        return "negativo", compound
    elif compound >= 0.35:
        return "positivo", compound
    else:
        return "neutral", compound


# ─────────────────────────────────────────────────────────────
#  VALIDACIÓN DE ENTRADA
# ─────────────────────────────────────────────────────────────

def validar_entrada(mensaje):
    """Verifica que el mensaje sea procesable."""
    if not mensaje or not mensaje.strip():
        return False, "empty"
    texto_limpio = re.sub(r"[^\w\s]", "", mensaje, flags=re.UNICODE).strip()
    if len(texto_limpio) < 2:
        return False, "only_symbols"
    if len(mensaje.strip()) < 2:
        return False, "too_short"
    return True, None


RESPUESTAS_INVALIDAS = {
    "empty":        "¡Hola! Parece que tu mensaje llegó vacío. ¿En qué te puedo ayudar? 😊",
    "only_symbols": "¡Hola! No entendí bien tu mensaje. ¿Puedes escribirme tu pregunta?",
    "too_short":    "¿Puedes contarme un poco más? Con gusto te ayudo 😊",
}


# ─────────────────────────────────────────────────────────────
#  CONSTRUCCIÓN DE PROMPTS
# ─────────────────────────────────────────────────────────────

def construir_prompt(intencion, confianza, datos, config, sentimiento):
    """Prompt dinámico según intención y sentimiento."""
    nombre_academia = config.get("nombre_academia", "Goku Lab")
    whatsapp = config.get("whatsapp", "")

    tono_map = {
        "negativo": "El usuario parece frustrado o molesto. Responde con mucha empatía, valida su sentimiento y sé especialmente paciente. No uses signos de exclamación excesivos.",
        "positivo": "El usuario está animado o entusiasmado. Mantén esa energía positiva y responde con entusiasmo.",
        "neutral":  "Responde de forma amable, clara y profesional.",
    }

    instrucciones_intencion = {
        "Saludo":                  f"El usuario está saludando. Salúdalo calurosamente, preséntate como el asistente virtual de {nombre_academia} y pregúntale en qué le puedes ayudar.",
        "Despedida":              (f"El usuario se está despidiendo. Despídete de forma amable e invítalo a regresar "
                                   f"cuando tenga más dudas sobre {nombre_academia}. "
                                   f"Termina SIEMPRE con exactamente esta frase: '¡Te esperamos en Goku Lab! 🎮\nJuega, Aprende y Emprende'"),
        "Desconocido":             f"No se pudo entender con claridad la consulta (confianza baja: {confianza:.0%}). Discúlpate amablemente y pídele que reformule su pregunta.",
        "Consultar_Cursos":        "El usuario pregunta por los cursos disponibles. Menciona los cursos con nombre, descripción breve y edad dirigida. Redacta de forma conversacional.",
        "Consultar_Costos":        "El usuario pregunta por los precios. Menciona el costo de cada curso con su moneda y opciones de pago si las hay.",
        "Consultar_Horarios":      "El usuario pregunta por los horarios. Preséntelos de forma clara por curso.",
        "Consultar_Ubicacion":     "El usuario pregunta por la ubicación. Da la dirección, referencias y link de Google Maps si está disponible.",
        "Consultar_Modalidad":     "El usuario pregunta si las clases son presenciales, virtuales o mixtas. Explica la modalidad de cada curso.",
        "Consultar_Certificacion": "El usuario pregunta si otorgan certificado o diploma. Responde con la información disponible.",
        "Consultar_ClaseDemo":     "El usuario pregunta por una clase de prueba. Explica cómo funciona la masterclass o clase demo.",
        "Consultar_FormasPago":    "El usuario pregunta por formas de pago. Menciona los métodos y si hay opción de abonos.",
        "Consultar_RequisitosEdad":"El usuario pregunta por requisitos de edad. Explica el rango de edad de cada curso.",
        "Consultar_Duracion":      "El usuario pregunta cuánto duran los cursos. Menciona la duración de cada uno.",
    }

    instruccion_intencion = instrucciones_intencion.get(
        intencion,
        f"Intención: {intencion} (confianza: {confianza:.0%}). Usa los datos disponibles para responder."
    )

    return f"""
Eres un asistente virtual amable de la academia {nombre_academia}.
Siempre respondes en español mexicano, de forma natural y concisa.

TONO: {tono_map.get(sentimiento, tono_map["neutral"])}

TAREA: {instruccion_intencion}

DATOS DISPONIBLES: {datos}

REGLAS IMPORTANTES:
- No inventes información que no esté en los datos proporcionados.
- No menciones que eres una IA a menos que el usuario te lo pregunte directamente.
- Sé MUY conciso: máximo 2 oraciones cortas. Si necesitas dar más info, da lo más importante y pregunta si quiere saber más
- No uses listas con viñetas; redacta de forma conversacional.
- No repitas el saludo si ya lo hiciste antes en la conversación.
- Termina siempre con una pregunta para seguir la conversación y convencer al usuario.
""".strip()


def construir_prompt_rag(contexto_pdf, config, sentimiento):
    """Prompt para cuando el clasificador no reconoce la intención (fallback RAG)."""
    nombre_academia = config.get("nombre_academia", "Goku Lab")
    whatsapp = config.get("whatsapp", "")

    tono_map = {
        "negativo": "El usuario parece frustrado. Responde con mucha empatía y paciencia.",
        "positivo": "El usuario está animado. Mantén esa energía positiva.",
        "neutral":  "Responde de forma amable, clara y profesional.",
    }

    return f"""
Eres un asistente virtual amable de la academia {nombre_academia}.
Siempre respondes en español mexicano, de forma natural y concisa.

TONO: {tono_map.get(sentimiento, tono_map["neutral"])}

TAREA: El usuario hizo una pregunta que no pudo clasificarse con certeza.
Usa únicamente la siguiente información de la academia para responder:

INFORMACIÓN DE LA ACADEMIA:
{contexto_pdf}

REGLAS IMPORTANTES:
- Responde solo con lo que está en la información proporcionada.
- Si la información no es suficiente, dile amablemente que no tienes ese dato
  e invítalo a contactar por WhatsApp: {whatsapp}.
- No menciones que eres una IA a menos que te lo pregunten.
- Sé conciso y termina con una pregunta para continuar la conversación.
""".strip()


# ─────────────────────────────────────────────────────────────
#  RESPUESTA DE EMERGENCIA
# ─────────────────────────────────────────────────────────────
RESPUESTA_FALLBACK = (
    "En este momento tengo un problema técnico. "
    "Por favor, intenta de nuevo en un momento o escríbenos directamente por WhatsApp. 🙏"
)

def llamar_groq(messages):
    """Intenta con cada API key hasta que una funcione."""
    for key in GROQ_KEYS:
        try:
            cliente = Groq(api_key=key)
            respuesta = cliente.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=256,
                temperature=0.7,
                messages=messages
            )
            return respuesta.choices[0].message.content
        except Exception as e:
            print(f"Key falló: {e}. Intentando siguiente...")
            continue
    return RESPUESTA_FALLBACK


# ─────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/logo.png")
def logo():
    return send_file("logo.png")

@app.route("/")
def index():
    return send_file("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Body JSON requerido"}), 400

        mensaje = data.get("mensaje", "").strip()
        numero = data.get("numero", "anonimo")

        # ── 1. Validación de entrada ──────────────────────────
        es_valido, motivo = validar_entrada(mensaje)
        if not es_valido:
            return jsonify({
                "respuesta": RESPUESTAS_INVALIDAS.get(motivo, "¿En qué te puedo ayudar?"),
                "intencion": "invalido",
                "sentimiento": None,
            }), 200

        # ── 2. Análisis de sentimiento ────────────────────────
        sentimiento, score_sentimiento = analizar_sentimiento(mensaje)

        # ── 3. Clasificación de intención ─────────────────────
        intencion, confianza = predecir_intent(mensaje)

        # ── 4. MongoDB o RAG según confianza ──────────────────
        usar_rag = intencion == "Desconocido"

        if usar_rag:
            datos = obtener_datos_por_intencion("Desconocido")
        else:
            datos = obtener_datos_por_intencion(intencion)

        config = datos.get("config") or {}

        # ── 5. Historial reciente del usuario (últimos 5) ─────
        historial_groq = []
        if coleccion is not None:
            historial_db = list(
                coleccion.find({"numero": numero}, {"_id": 0, "mensaje": 1, "respuesta": 1})
                .sort("timestamp", -1)
                .limit(4)
            )
            for h in reversed(historial_db):
                historial_groq.append({"role": "user",      "content": h["mensaje"]})
                historial_groq.append({"role": "assistant", "content": h["respuesta"]})

        # ── 6. Prompt dinámico ────────────────────────────────
        if usar_rag and CONTEXTO_PDF:
            prompt_sistema = construir_prompt_rag(CONTEXTO_PDF, config, sentimiento)
        else:
            prompt_sistema = construir_prompt(intencion, confianza, datos, config, sentimiento)

        # ── 7. Llamada a Groq ─────────────────────────────────
        respuesta = llamar_groq([{"role": "system", "content": prompt_sistema},*historial_groq,{"role": "user", "content": mensaje},
])

        # ── 8. Guardar en MongoDB ─────────────────────────────
        if coleccion is not None:
            try:
                coleccion.insert_one({
                    "numero":      numero,
                    "mensaje":     mensaje,
                    "intencion":   intencion,
                    "confianza":   round(confianza, 4),
                    "sentimiento": sentimiento,
                    "score_sent":  round(score_sentimiento, 4),
                    "uso_rag":     usar_rag,
                    "respuesta":   respuesta,
                    "timestamp":   datetime.now(),
                })
            except Exception as mongo_err:
                print(f"No se pudo guardar en MongoDB: {mongo_err}")

        # ── 9. Respuesta al cliente ───────────────────────────
        return jsonify({
            "intencion":   intencion,
            "confianza":   f"{confianza:.0%}",
            "sentimiento": sentimiento,
            "respuesta":   respuesta,
        })

    except Exception as e:
        print(f"Error inesperado en /chat: {e}")
        return jsonify({"respuesta": RESPUESTA_FALLBACK}), 200


@app.route("/retrain", methods=["POST"])
def retrain():
    """Reentrenar el modelo manualmente sin tocar código."""
    global mejor_modelo, vectorizer
    try:
        if os.path.exists("intencione.xlsx"):
            os.remove("intencione.xlsx")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        mejor_modelo, vectorizer = entrenar_y_guardar()
        return jsonify({"status": "ok", "mensaje": "Modelo reentrenado exitosamente"}), 200
    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "modelo_cargado": mejor_modelo is not None,
        "mongo_ok":       db is not None,
        "groq_ok": len(GROQ_KEYS) > 0,
        "rag_listo":      bool(CONTEXTO_PDF),
        "timestamp":      datetime.now().isoformat(),
    }), 200


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Arrancando Flask en puerto {port}...")
    app.run(host="0.0.0.0", port=port)

