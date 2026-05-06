import os
import gdown
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("punkt_tab")
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import os
import pandas as pd
import gdown

file_id = "1viVnkIq_QIp8jI_Ysye6Q_WVerPsvUvE"
file_name = "intencione.xlsx"
url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"

if not os.path.exists(file_name):
    print("Descargando archivo...")
    gdown.download(url, file_name, quiet=False)
else:
    print(f"El archivo '{file_name}' ya está descargado")

df = pd.read_excel(file_name, engine="openpyxl")

print(df.head())

df.head()

df = df.drop(columns = ["Marca temporal","Dirección de correo electrónico"],axis=1)

df_final= pd.melt(df, value_vars= df.columns, var_name= "Intent",value_name="Texto")

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

df_final

df_final.isnull()
df_final[df_final.isnull().any(axis=1)]


df_final = df_final.dropna(subset=["Texto"])



"""# **preprocesamiento**"""

import re
import unicodedata
import string

def limpiar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))

    # texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

df_final['Texto'] = df_final['Texto'].apply(limpiar_texto)

print(df_final)

stop_words = set(stopwords.words("spanish"))

sin_stop_words = []
for oracion in df_final["Texto"]:
    words = [pal for pal in oracion.split() if pal not in stop_words]
    sin_stop_words.append(" ".join(words))

df_final["Texto"] = sin_stop_words
sin_stop_words

df_final

df_final["Tokens"] = df_final["Texto"].apply(nltk.word_tokenize)

df_final

"""# **TF IDF**"""

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_final["Texto"])
Y = df_final["Intent"]

terminos = vectorizer.get_feature_names_out()
tabla = pd.DataFrame(
    X.todense(),
    columns=terminos,
    index=[f"Registo {i+1}" for i in range(len(df_final["Texto"]))]
)
tabla

"""# **Entrenar el modelo**"""

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=123, stratify=Y)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train, Y_train)

print("mejores parámetros:", grid_search.best_params_)
print("f1 macro en CV:", grid_search.best_score_.round(4))

mejor_modelo = grid_search.best_estimator_
Y_pred = mejor_modelo.predict(X_test)
print(classification_report(Y_test, Y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

Y_pred = mejor_modelo.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=mejor_modelo.classes_
)

fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout()
plt.show()

nuevas_entradas = [
    "¿Dónde se toman las clases?",
    "¿En qué lugar son las clases?",
    "¿Tengo que ir a algún lado o es en línea?",
    "¿Las clases son en una sede o desde casa?",
    "¿En qué parte se dan las clases?",
    "¿A qué hora son las clases en línea?",
    "¿Las clases presenciales qué días son?",
    "¿Cómo son los horarios de las clases virtuales?",
    "¿Las clases online en qué horario son?",
    "¿Se conectan o hay que asistir?",
    "¿Cómo funcionan las clases?",
    "¿Cómo se toman las clases?",
    "¿Tengo que ir o todo es por compu?",
    "¿Es necesario presentarse?",
    "¿Cuánto duran las clases por día?",
    "¿Cuántas horas son a la semana?",
    "¿Qué tan largas son las sesiones?",
    "¿Cuánto tiempo es cada clase?",
    "¿Cuánto dura el curso por día?",
    "¿Cuánto tengo que pagar y cómo?",
    "¿Se puede pagar en partes?",
    "¿Cuál es el costo y si aceptan tarjeta?",
    "¿Hay pagos mensuales?",
    "¿Cuánto cuesta y cómo se paga?"
]

nuevas_limpias = [limpiar_texto(texto) for texto in nuevas_entradas]

nuevas_vectorizadas = vectorizer.transform(nuevas_limpias)

predicciones = mejor_modelo.predict(nuevas_vectorizadas)

for texto, intent in zip(nuevas_entradas, predicciones):
    print(f"'{texto}' → {intent}")


#nota: después que el modelo quedo entrenada, agregamos la conexión a mongodb
#flask para que reciba los mensajes
#una función que clasifica y además guarda en MONGODB

from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime


client = MongoClient("mongodb://localhost:27017/")
db = client["academia_chatbot"]
coleccion = db["conversaciones"]

respuestas = {
    "Consultar_Ubicacion": "Estamos ubicados en [dirección de la academia].",
    "Consultar_Costos": "Los costos varían según el curso. ¿Te interesa alguno en particular?",
    "Consultar_Horarios": "Manejamos horarios matutinos y vespertinos. ¿Cuál te acomoda?",
    "Consultar_Certificacion": "Sí, otorgamos certificados al completar cada curso.",
    "Saludo": "¡Hola! Bienvenido a la academia. ¿En qué te puedo ayudar?",
    "Despedida": "¡Hasta luego! Fue un placer ayudarte.",
    "Consultar_Modalidad": "Ofrecemos clases presenciales, virtuales y mixtas.",
    "Consultar_Cursos": "Tenemos cursos de [lista de cursos]. ¿Te interesa alguno?",
    "Consultar_ClaseDemo": "Sí, ofrecemos una clase demo gratuita. ¿Quieres agendar una?",
    "Consultar_RequisitosEdad": "No hay restricción de edad para tomar nuestros cursos.",
    "Consultar_FormasPago": "Aceptamos efectivo, tarjeta y transferencia bancaria.",
    "Consultar_Duracion": "La duración depende del curso, generalmente entre 4 y 12 semanas.",
}

app = Flask(__name__)
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    mensaje = data.get("mensaje", "")

   
    mensaje_limpio = limpiar_texto(mensaje)
    mensaje_vectorizado = vectorizer.transform([mensaje_limpio])
    intencion = mejor_modelo.predict(mensaje_vectorizado)[0]

    respuesta = respuestas.get(intencion, "No entendí tu pregunta, ¿puedes reformularla?")

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


