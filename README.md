# Chatbot Goku Lab

Chatbot para la academia **Goku Lab**, desarrollado como prácticas profesionales en Entremex


## Arquitectura

```
Usuario → Validación → Análisis de sentimiento → Clasificador TF-IDF + SVC
       → MongoDB Atlas (datos de la academia)
       → Groq API / LLaMA 3.3 70B (generación de respuesta)
       → Fallback RAG con PDF si el clasificador no reconoce la intención
```

## Características

- Clasificador de intenciones con TF-IDF + SVC (~90% de precisión)
- 12 intenciones: cursos, costos, horarios, ubicación, modalidad, certificación, clase demo, formas de pago, requisitos de edad, duración, saludo y despedida
- Análisis de sentimiento con VADER (positivo/negativo/neutral)
- Respuestas generadas con LLaMA 3.3 70B vía Groq API
- Fallback RAG con pdfplumber cuando el clasificador no reconoce la consulta
- Historial de conversación por usuario en MongoDB Atlas
- Interfaz web de chat incluida
- Soporte para múltiples API keys de Groq como respaldo

---

## Estructura del proyecto

```
chatbot_gokulab_valeria/
├── chatbot_v2.py        # Backend principal (Flask)
├── chat.html            # Interfaz web de chat
├── gokulab_info.pdf     # Base de conocimiento para RAG
├── modelo_intents.pkl   # Modelo entrenado (TF-IDF + SVC)
├── logo.png             # Logo de Goku Lab
├── requirements.txt     # Dependencias
└── .gitignore
```

---

## Instalación y uso local

### 1. Clonar el repositorio
```bash
git clone https://github.com/ValeriaDeita/chatbot_gokulab_valeria.git
cd chatbot_gokulab_valeria
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Crear archivo `.env`
Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```
MONGO_URI=tu_uri_de_mongodb_atlas
GROQ_API_KEY_1=tu_primera_api_key_de_groq
GROQ_API_KEY_2=tu_segunda_api_key_de_groq
----
son 5 api keys para pruebas
```

> Las API keys de Groq son gratuitas en https://console.groq.com

### 4. Correr el servidor
```bash
python chatbot_v2.py
```

### 5. Abrir la interfaz
Abre `chat.html` en tu navegador o entra a:
```
http://127.0.0.1:5000
```

---

## Despliegue en producción

El chatbot está desplegado en Render:
```
https://chatbot-gokulab-valeria.onrender.com
```



Desarrollado por: 
**Valeria Deita Rosario**  
