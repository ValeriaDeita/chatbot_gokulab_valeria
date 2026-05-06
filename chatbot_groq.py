import os
import re
import unicodedata
import string
import gdown
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from flask import Flask, request, jsonify
from pymongo import MongoClient
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

nltk.download("stopwords")
nltk.download("punkt_tab")
load_dotenv()

client_mongo = MongoClient(os.getenv("MONGO_URI"))
db = client_mongo["chatbot_Goku_lab"]
coleccion = db["conversaciones"]
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


file_id = "1viVnkIq_QIp8jI_Ysye6Q_WVerPsvUvE"
file_name = "intencione.xlsx"
url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"

if not os.path.exists(file_name):
    gdown.download(url, file_name, quiet=False)

df = pd.read_excel(file_name, engine="openpyxl")
df = df.drop(columns=["Marca temporal", "Dirección de correo electrónico"], axis=1)
df_final = pd.melt(df, value_vars=df.columns, var_name="Intent", value_name="Texto")
df_final["Intent"] = df_final["Intent"].str.strip()
df_final["Intent"] = df_final["Intent"].replace({
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
    "Escribe cómo preguntarías la duración de los cursos": "Consultar_Duracion"
})
df_final = df_final.dropna(subset=["Texto"])

def limpiar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

df_final['Texto'] = df_final['Texto'].apply(limpiar_texto)
stop_words = set(stopwords.words("spanish"))
df_final["Texto"] = df_final["Texto"].apply(
    lambda oracion: " ".join([p for p in oracion.split() if p not in stop_words])
)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_final["Texto"])
Y = df_final["Intent"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
grid_search = GridSearchCV(
    SVC(), {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
    cv=cv, scoring='f1_macro', n_jobs=-1
)
grid_search.fit(X, Y)
mejor_modelo = grid_search.best_estimator_
print("Modelo entrenado!")

def obtener_datos_por_intencion(intencion):
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
        return {"direccion": config.get("direccion"), "referencias": config.get("referencias"), "config": config}
    else:
        return {"config": config}

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    mensaje = data.get("mensaje", "")

    mensaje_limpio = limpiar_texto(mensaje)
    mensaje_vectorizado = vectorizer.transform([mensaje_limpio])
    intencion = mejor_modelo.predict(mensaje_vectorizado)[0]

    datos = obtener_datos_por_intencion(intencion)
    config = datos.get("config") or {}

    prompt_sistema = f"""
    Eres un asistente amable de la academia {config.get('nombre_academia', 'GokuLab')}.
    Responde de forma natural, amable y en español mexicano.
    La intención del usuario es: {intencion}
    Datos relevantes de la academia: {datos}
    Si no tienes información suficiente, invita al usuario a pedir mas info {config.get('whatsapp', '')}.
    """

    respuesta_groq = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": mensaje}
        ]
    )
    respuesta = respuesta_groq.choices[0].message.content

    coleccion.insert_one({
        "mensaje": mensaje,
        "intencion": intencion,
        "respuesta": respuesta,
        "timestamp": datetime.now()
    })

    return jsonify({
        "intencion": intencion,
        "respuesta": respuesta
    })

if __name__ == "__main__":
    app.run(debug=True)