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
from flask import Flask, request, jsonify
from pymongo import MongoClient
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file



# ─────────────────────────────────────────────────────────────
#  SETUP INICIAL
# ─────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
load_dotenv()

# ── Conexiones ────────────────────────────────────────────────
try:
    client_mongo = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
    client_mongo.server_info()  # Verifica conexión al arrancar
    db = client_mongo["chatbot_Goku_lab"]
    coleccion = db["conversaciones"]
    print("✅ MongoDB conectado.")
except Exception as e:
    print(f"❌ Error conectando a MongoDB: {e}")
    db = None
    coleccion = None

try:
    client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("✅ Groq conectado.")
except Exception as e:
    print(f"❌ Error conectando a Groq: {e}")
    client_groq = None

# ── Analizador de sentimiento ─────────────────────────────────
analizador_sentimiento = SentimentIntensityAnalyzer()

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
        print("📥 Descargando dataset...")
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

    print(f"✅ Modelo entrenado. Mejor config: {gs.best_params_}")
    return gs.best_estimator_, vec


def cargar_modelo():
    """Carga el modelo desde disco o entrena uno nuevo si no existe."""
    if os.path.exists(MODEL_PATH):
        print("✅ Modelo cargado desde disco.")
        with open(MODEL_PATH, "rb") as f:
            datos = pickle.load(f)
        return datos["modelo"], datos["vectorizer"]
    print("⚙️  No se encontró modelo en disco. Entrenando...")
    return entrenar_y_guardar()


# ── Cargar modelo al arrancar (con manejo de error) ───────────
try:
    mejor_modelo, vectorizer = cargar_modelo()
except Exception as e:
    print(f"❌ Error cargando/entrenando modelo: {e}")
    mejor_modelo, vectorizer = None, None


# ─────────────────────────────────────────────────────────────
#  UTILIDADES DE CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────

def predecir_intent(texto, umbral=0.5):
    """
    Clasifica el texto. Si la confianza es menor al umbral,
    devuelve 'Desconocido' para activar la respuesta de fallback.
    """
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
        return {
            "cursos": list(db["cursos"].find({}, {"_id": 0, "nombreCurso": 1, "precio": 1, "moneda": 1})),
            "config": config,
        }

    elif intencion == "Consultar_Horarios":
        return {"horarios": list(db["horarios"].find({}, {"_id": 0})), "config": config}

    elif intencion == "Consultar_Certificacion":
        return {"certificacion": config.get("certificacion"), "config": config}

    elif intencion == "Consultar_ClaseDemo":
        return {"masterclass": config.get("masterclass"), "config": config}

    elif intencion == "Consultar_FormasPago":
        return {
            "pagos": config.get("formas_pago"),
            "abonos": config.get("detalle_abonos"),
            "config": config,
        }

    elif intencion == "Consultar_Modalidad":
        return {
            "cursos": list(db["cursos"].find({}, {"_id": 0, "nombreCurso": 1, "modalidad": 1})),
            "config": config,
        }

    elif intencion == "Consultar_RequisitosEdad":
        return {
            "cursos": list(db["cursos"].find({}, {"_id": 0, "nombreCurso": 1, "edad_dirigida": 1})),
            "config": config,
        }

    elif intencion == "Consultar_Duracion":
        return {
            "cursos": list(db["cursos"].find({}, {"_id": 0, "nombreCurso": 1, "duracion_min_por_clase": 1})),
            "config": config,
        }

    elif intencion == "Consultar_Ubicacion":
        return {
            "direccion": config.get("direccion"),
            "referencias": config.get("referencias"),
            "link_maps": config.get("link_maps"),
            "config": config,
        }

    # Saludo, Despedida, Desconocido → solo config (el prompt se encarga)
    return {"config": config}


# ─────────────────────────────────────────────────────────────
#  ANÁLISIS DE SENTIMIENTO
# ─────────────────────────────────────────────────────────────

def analizar_sentimiento(texto):
    """
    Usa VADER para detectar el tono emocional del mensaje.
    Funciona con español básico (positivo/negativo/neutral).
    Retorna: (etiqueta, score_compound)
    """
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
    """
    Verifica que el mensaje sea procesable.
    Retorna: (es_valido: bool, motivo: str | None)
    """
    if not mensaje or not mensaje.strip():
        return False, "empty"

    # Quitar emojis y símbolos; si no queda texto real, rechazar
    texto_limpio = re.sub(r"[^\w\s]", "", mensaje, flags=re.UNICODE).strip()
    if len(texto_limpio) < 2:
        return False, "only_symbols"

    # Mensaje demasiado corto para clasificar con sentido
    if len(mensaje.strip()) < 3:
        return False, "too_short"

    return True, None


RESPUESTAS_INVALIDAS = {
    "empty":        "¡Hola! Parece que tu mensaje llegó vacío. ¿En qué te puedo ayudar? 😊",
    "only_symbols": "¡Hola! No entendí bien tu mensaje. ¿Puedes escribirme tu pregunta con palabras?",
    "too_short":    "¿Puedes contarme un poco más? Con gusto te ayudo 😊",
}


# ─────────────────────────────────────────────────────────────
#  CONSTRUCCIÓN DE PROMPT DINÁMICO
# ─────────────────────────────────────────────────────────────

def construir_prompt(intencion, confianza, datos, config, sentimiento):
    """
    Genera el system prompt para Groq adaptando:
    - La intención detectada (qué responder)
    - El sentimiento del usuario (cómo responderlo)
    """
    nombre_academia = config.get("nombre_academia", "Goku Lab")
    whatsapp = config.get("whatsapp", "")

    # ── Tono según sentimiento ────────────────────────────────
    tono_map = {
        "negativo": (
            "El usuario parece frustrado o molesto. "
            "Responde con mucha empatía, valida su sentimiento y sé especialmente paciente. "
            "No uses signos de exclamación excesivos."
        ),
        "positivo": (
            "El usuario está animado o entusiasmado. "
            "Mantén esa energía positiva y responde con entusiasmo."
        ),
        "neutral": "Responde de forma amable, clara y profesional.",
    }
    instruccion_tono = tono_map.get(sentimiento, tono_map["neutral"])

    # ── Instrucción por intención ─────────────────────────────
    instrucciones_intencion = {
        "Saludo": (
            f"El usuario está saludando. Salúdalo calurosamente, preséntate como el asistente virtual "
            f"de {nombre_academia} y pregúntale en qué le puedes ayudar."
        ),
        "Despedida": (
            f"El usuario se está despidiendo. Despídete de forma amable e invítalo a regresar "
            f"cuando tenga más dudas sobre {nombre_academia}."
        ),
        "Desconocido": (
            f"No se pudo entender con claridad la consulta del usuario (confianza baja: {confianza:.0%}). "
            f"Discúlpate amablemente, dile que no entendiste bien su pregunta y pídele que la reformule. "
            f"Si persiste la duda, invítalo a escribir directamente a WhatsApp: {whatsapp}."
        ),
        "Consultar_Cursos": (
            "El usuario pregunta por los cursos disponibles. "
            "Menciona los cursos con su nombre, breve descripción y para qué edad van dirigidos. "
            "Sé organizado pero no uses listas de puntos largas; redacta de forma conversacional."
        ),
        "Consultar_Costos": (
            "El usuario pregunta por los precios. "
            "Menciona el costo de cada curso con su moneda. "
            "Si hay opciones de pago o abonos, menciónalo brevemente."
        ),
        "Consultar_Horarios": (
            "El usuario pregunta por los horarios disponibles. "
            "Presenta los horarios de forma clara, por curso si aplica."
        ),
        "Consultar_Ubicacion": (
            "El usuario pregunta por la ubicación de la academia. "
            "Da la dirección, referencias útiles y el link de Google Maps si está disponible."
        ),
        "Consultar_Modalidad": (
            "El usuario pregunta si las clases son presenciales, virtuales o mixtas. "
            "Explica la modalidad de cada curso de forma clara."
        ),
        "Consultar_Certificacion": (
            "El usuario pregunta si la academia otorga certificado o diploma. "
            "Responde con la información disponible sobre certificación."
        ),
        "Consultar_ClaseDemo": (
            "El usuario pregunta si puede tomar una clase de prueba o demo antes de inscribirse. "
            "Explica cómo funciona la masterclass o clase demo si está disponible."
        ),
        "Consultar_FormasPago": (
            "El usuario pregunta por las formas de pago aceptadas. "
            "Menciona los métodos de pago y si hay posibilidad de pagar en abonos."
        ),
        "Consultar_RequisitosEdad": (
            "El usuario pregunta si hay restricciones de edad para los cursos. "
            "Explica el rango de edad al que va dirigido cada curso."
        ),
        "Consultar_Duracion": (
            "El usuario pregunta cuánto dura cada curso o cada clase. "
            "Menciona la duración de cada curso de forma clara."
        ),
    }

    instruccion_intencion = instrucciones_intencion.get(
        intencion,
        f"Intención detectada: {intencion} (confianza: {confianza:.0%}). "
        f"Usa los datos disponibles para responder de forma natural y completa."
    )

    return f"""
Eres un asistente virtual amable de la academia {nombre_academia}.
Siempre respondes en español mexicano, de forma natural y concisa.

TONO: {instruccion_tono}

TAREA: {instruccion_intencion}

DATOS DISPONIBLES: {datos}

REGLAS IMPORTANTES:
- No inventes información que no esté en los datos proporcionados.
- Si no tienes el dato exacto, invita al usuario a preguntar directamente al WhatsApp: {whatsapp}.
- No menciones que eres una IA a menos que el usuario te lo pregunte directamente.
- Sé conciso: máximo 3-4 oraciones, salvo que la información requiera más detalle.
- No uses listas con viñetas; redacta de forma conversacional.
- No repitas el saludo si ya lo hiciste antes en la conversación.
""".strip()


# ─────────────────────────────────────────────────────────────
#  RESPUESTA DE EMERGENCIA (cuando Groq falla)
# ─────────────────────────────────────────────────────────────

RESPUESTA_FALLBACK = (
    "En este momento tengo un problema técnico. "
    "Por favor, intenta de nuevo en un momento o escríbenos directamente por WhatsApp. 🙏"
)


# ─────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
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
        numero = data.get("numero", "anonimo")  # número de WhatsApp del usuario

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

        # ── 4. Consulta a MongoDB ─────────────────────────────
        datos = obtener_datos_por_intencion(intencion)
        config = datos.get("config") or {}

        # ── 5. Historial reciente del usuario (últimos 5) ─────
        historial_groq = []
        if coleccion is not None:
            historial_db = list(
                coleccion.find({"numero": numero}, {"_id": 0, "mensaje": 1, "respuesta": 1})
                .sort("timestamp", -1)
                .limit(5)
            )
            for h in reversed(historial_db):
                historial_groq.append({"role": "user",      "content": h["mensaje"]})
                historial_groq.append({"role": "assistant", "content": h["respuesta"]})

        # ── 6. Construcción del prompt dinámico ───────────────
        prompt_sistema = construir_prompt(intencion, confianza, datos, config, sentimiento)

        # ── 7. Llamada a Groq ─────────────────────────────────
        if client_groq is None:
            return jsonify({"respuesta": RESPUESTA_FALLBACK}), 200

        try:
            respuesta_groq = client_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=512,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    *historial_groq,
                    {"role": "user", "content": mensaje},
                ],
            )
            respuesta = respuesta_groq.choices[0].message.content
        except Exception as groq_err:
            print(f"❌ Error en Groq: {groq_err}")
            respuesta = RESPUESTA_FALLBACK

        # ── 8. Guardar conversación en MongoDB ────────────────
        if coleccion is not None:
            try:
                coleccion.insert_one({
                    "numero":      numero,
                    "mensaje":     mensaje,
                    "intencion":   intencion,
                    "confianza":   round(confianza, 4),
                    "sentimiento": sentimiento,
                    "score_sent":  round(score_sentimiento, 4),
                    "respuesta":   respuesta,
                    "timestamp":   datetime.now(),
                })
            except Exception as mongo_err:
                print(f"⚠️  No se pudo guardar en MongoDB: {mongo_err}")
                # No interrumpimos la respuesta al usuario por esto

        # ── 9. Respuesta al cliente ───────────────────────────
        return jsonify({
            "intencion":   intencion,
            "confianza":   f"{confianza:.0%}",
            "sentimiento": sentimiento,
            "respuesta":   respuesta,
        })

    except Exception as e:
        print(f"❌ Error inesperado en /chat: {e}")
        return jsonify({
            "respuesta": RESPUESTA_FALLBACK
        }), 200  # 200 para que WhatsApp no reintente el webhook


# ─────────────────────────────────────────────────────────────
#  ENDPOINT: REENTRENAR MANUALMENTE (parametrizable)
#  POST /retrain  → recarga el modelo desde el dataset actualizado
#  Útil cuando agregan nuevas intenciones al dataset
# ─────────────────────────────────────────────────────────────
@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Endpoint para reentrenar el modelo sin tocar código.
    Llamar cuando se agreguen nuevas intenciones al dataset de Google Sheets.
    Los datos de MongoDB (cursos, precios, etc.) NO requieren reentrenamiento.
    """
    global mejor_modelo, vectorizer
    try:
        # Forzar re-descarga del dataset y reentrenamiento
        if os.path.exists("intencione.xlsx"):
            os.remove("intencione.xlsx")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        mejor_modelo, vectorizer = entrenar_y_guardar()
        return jsonify({"status": "ok", "mensaje": "Modelo reentrenado exitosamente ✅"}), 200
    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)}), 500


# ─────────────────────────────────────────────────────────────
#  ENDPOINT: HEALTH CHECK
# ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "modelo_cargado": mejor_modelo is not None,
        "mongo_ok":       db is not None,
        "groq_ok":        client_groq is not None,
        "timestamp":      datetime.now().isoformat(),
    }), 200


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

    