import pandas as pd
import re
from datetime import datetime
import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import plotly.express as px
import emoji
import string

# Descargar recursos de nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Inicializar herramientas NLP
lemmatizer = WordNetLemmatizer()
sentiment_analysis = pipeline('sentiment-analysis', model='dccuchile/bert-base-spanish-wwm-uncased')

# Función para detectar emojis
def contiene_emojis(texto):
    return any(char in emoji.EMOJI_DATA for char in texto)

# Función para limpiar texto (normalización + limpieza de caracteres especiales)
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)  # eliminar URLs
    texto = re.sub(r'\d+', '', texto)  # eliminar números
    texto = texto.translate(str.maketrans('', '', string.punctuation))  # eliminar puntuación
    texto = texto.strip()
    return texto

# Función para procesar el archivo .txt de WhatsApp
def cargar_chat_txt(file):
    content = file.getvalue().decode('utf-8')
    lines = content.splitlines()

    fechas = []
    autores = []
    mensajes = []

    pattern = r"(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}\s?[ap]\.\s?[m]\.) - (.*?):(.*)"

    for line in lines:
        if "cifrados de extremo a extremo" in line:
            continue

        match = re.match(pattern, line.strip())
        if match:
            fecha = match.group(1)
            hora = match.group(2)
            autor = match.group(3).strip()
            mensaje = match.group(4).strip()

            hora = hora.replace("\u202f", "").strip().replace(".", "")
            fecha_hora_str = f"{fecha} {hora}"

            try:
                fecha_hora = datetime.strptime(fecha_hora_str, "%d/%m/%Y %I:%M%p")
            except ValueError as e:
                print(f"Error al parsear la fecha y hora: {fecha_hora_str} - {e}")
                continue

            fechas.append(fecha_hora)
            autores.append(autor)
            mensajes.append(mensaje)

    df = pd.DataFrame({
        'FechaHora': fechas,
        'Autor': autores,
        'Mensaje': mensajes
    })

    if not df.empty:
        df['FechaHora'] = pd.to_datetime(df['FechaHora'])
        df['Mensaje'] = df['Mensaje'].apply(limpiar_texto)
        return df
    else:
        return None

# Quitar stopwords y aplicar lematización
def quitar_stopwords_lemmatizar(mensaje):
    stop_words = set(stopwords.words('spanish'))
    tokens = word_tokenize(mensaje.lower())
    tokens_filtrados = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens_filtrados)

# Extraer bigramas y trigramas
def extraer_bigrams_trigrams(mensaje):
    tokens = word_tokenize(mensaje.lower())
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))
    return bigrams, trigrams

# Funciones de urgencia (autor, hora, sentimiento, palabras clave, etc.)
def urgencia_por_autor(autor):
    autores_prioritarios = ["Jefe", "Hijo", "Mamá", "Papá", "Esposa", "Novia"]
    if any(char in autor for char in ["❤️", "💖", "💘", "💝", "💕"]):
        return 2
    return 2 if autor in autores_prioritarios else 0

def urgencia_por_hora(hora):
    hora = datetime.strptime(hora, "%H:%M")
    if hora >= datetime.strptime("20:00", "%H:%M") or hora <= datetime.strptime("05:00", "%H:%M"):
        return 1
    return 0

def urgencia_por_sentimiento(sentimiento):
    etiquetas = {'LABEL_4': 3, 'LABEL_3': 2, 'LABEL_2': 1, 'LABEL_1': 1, 'LABEL_0': 0}
    return etiquetas.get(sentimiento, 0)

def urgencia_por_palabras_clave(mensaje):
    claves = ["urgente", "es urgente", "es para hoy", "necesito ayuda", "por favor", "con urgencia", "rapido", "callo", "caer", "atropellado"]
    mensaje = mensaje.lower()
    return 1 if any(clave in mensaje for clave in claves) else 0

def urgencia_por_palabras_negativas(mensaje):
    negativas = ["malo", "no me gusta", "odio", "peor", "terrible", "desastroso", "fatal"]
    mensaje = mensaje.lower()
    return 2 if any(p in mensaje for p in negativas) else 0

def mapear_sentimiento(sentimiento):
    mapeo = {'LABEL_0': 'Muy Positivo', 'LABEL_1': 'Positivo', 'LABEL_2': 'Neutro', 'LABEL_3': 'Negativo', 'LABEL_4': 'Muy Negativo'}
    return mapeo.get(sentimiento, 'Desconocido')

def calcular_urgencia(row):
    mensaje_filtrado = quitar_stopwords_lemmatizar(row['Mensaje'])
    bigrams, trigrams = extraer_bigrams_trigrams(mensaje_filtrado)
    result = sentiment_analysis(mensaje_filtrado)
    sentimiento = result[0]['label']
    probabilidad = result[0]['score']
    sentimiento_legible = mapear_sentimiento(sentimiento)
    if probabilidad < 0.6:
        sentimiento_legible = 'Neutro'

    urgencia = (
        urgencia_por_autor(row['Autor']) +
        urgencia_por_hora(row['FechaHora'].strftime('%H:%M')) +
        urgencia_por_sentimiento(sentimiento) +
        urgencia_por_palabras_clave(row['Mensaje']) +
        urgencia_por_palabras_negativas(row['Mensaje'])
    )

    # Añadir urgencia si contiene emojis emocionales
    if contiene_emojis(row['Mensaje']):
        urgencia += 1

    for bigram in bigrams + trigrams:
        bigram_str = ' '.join(bigram)
        sentimiento_bigram = sentiment_analysis(bigram_str)[0]['label']
        urgencia += urgencia_por_sentimiento(sentimiento_bigram)

    return min(5, urgencia), sentimiento_legible

# Funciones para mostrar bigramas y trigramas
def obtener_bigrams_trigrams(df):
    bigramas, trigramas = [], []
    for mensaje in df['Mensaje']:
        bigs, trigs = extraer_bigrams_trigrams(mensaje)
        bigramas.extend([' '.join(b) for b in bigs])
        trigramas.extend([' '.join(t) for t in trigs])
    return bigramas, trigramas

def mostrar_grafica_bigrams_trigrams(bigramas, trigramas):
    bigram_df = pd.DataFrame(Counter(bigramas).most_common(10), columns=['Bigram', 'Frecuencia'])
    trigram_df = pd.DataFrame(Counter(trigramas).most_common(10), columns=['Trigram', 'Frecuencia'])
    return px.bar(bigram_df, x='Bigram', y='Frecuencia', title="Top 10 Bigramas"), px.bar(trigram_df, x='Trigram', y='Frecuencia', title="Top 10 Trigramas")

# App de Streamlit
st.title("Análisis de Chat de WhatsApp con Urgencia y Emojis")
uploaded_file = st.file_uploader("Sube un archivo TXT de chat de WhatsApp", type=["txt"])

if uploaded_file is not None:
    df_chat = cargar_chat_txt(uploaded_file)
    if df_chat is not None and not df_chat.empty:
        st.dataframe(df_chat.head())
        df_chat[['Urgencia', 'Sentimiento']] = df_chat.apply(calcular_urgencia, axis=1, result_type='expand')
        st.dataframe(df_chat[['FechaHora', 'Autor', 'Mensaje', 'Urgencia', 'Sentimiento']])

        bigramas, trigramas = obtener_bigrams_trigrams(df_chat)
        fig_bigram, fig_trigram = mostrar_grafica_bigrams_trigrams(bigramas, trigramas)
        st.plotly_chart(fig_bigram)
        st.plotly_chart(fig_trigram)

        if st.button('Mostrar Gráfico de Urgencia'):
            urgencia_count = df_chat['Urgencia'].value_counts().reset_index()
            urgencia_count.columns = ['Urgencia', 'Cantidad']
            fig = px.bar(urgencia_count, x='Urgencia', y='Cantidad', title="Distribución de Urgencia")
            st.plotly_chart(fig)
    else:
        st.write("No se encontraron mensajes válidos.")
