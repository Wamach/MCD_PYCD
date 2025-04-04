import pandas as pd
import re
from datetime import datetime
import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import plotly.express as px

# Descargar stopwords de nltk
nltk.download('stopwords')
nltk.download('punkt')

# Cargar el pipeline de análisis de sentimientos
sentiment_analysis = pipeline('sentiment-analysis', model='dccuchile/bert-base-spanish-wwm-uncased')

# Función para procesar el archivo .txt de WhatsApp
def cargar_chat_txt(file):
    content = file.getvalue().decode('utf-8')
    lines = content.splitlines()
    
    fechas = []
    autores = []
    mensajes = []

    pattern = r"(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}\s?[ap]\.?\s?[m]\.?) - (.*?):(.*)"
    
    for line in lines:
        if "cifrados de extremo a extremo" in line:
            continue
        
        match = re.match(pattern, line.strip())
        
        if match:
            fecha = match.group(1)
            hora = match.group(2)
            autor = match.group(3).strip()
            mensaje = match.group(4).strip()

            hora = hora.replace("\u202f", "").strip()
            hora = hora.replace(".", "")
            
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
    
    if 'FechaHora' in df.columns and 'Autor' in df.columns and 'Mensaje' in df.columns:
        df['FechaHora'] = pd.to_datetime(df['FechaHora'])
        return df
    else:
        return None

# Función para quitar las stopwords de los mensajes
def quitar_stopwords(mensaje):
    stop_words = set(stopwords.words('spanish'))
    mensaje_tokens = nltk.word_tokenize(mensaje.lower())
    mensaje_filtrado = [word for word in mensaje_tokens if word not in stop_words]
    return ' '.join(mensaje_filtrado)

# Función para extraer bigramas y trigramas
def extraer_bigrams_trigrams(mensaje):
    tokens = nltk.word_tokenize(mensaje.lower())
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))
    return bigrams, trigrams

# Función para clasificar la urgencia basada en el autor
def urgencia_por_autor(autor):
    autores_prioritarios = ["Jefe", "Hijo", "Mamá", "Papá", "Esposa"]
    
    if any(char in autor for char in ["❤️", "💖", "💘", "💝", "💕"]):
        return 2  # Asignar urgencia alta si el autor tiene corazón en su nombre
    
    return 2 if autor in autores_prioritarios else 0

# Función para clasificar la urgencia basada en la hora
def urgencia_por_hora(hora):
    hora = datetime.strptime(hora, "%H:%M")
    if hora >= datetime.strptime("20:00", "%H:%M") or hora <= datetime.strptime("05:00", "%H:%M"):
        return 1
    return 0

# Función para clasificar la urgencia basada en el sentimiento
def urgencia_por_sentimiento(sentimiento):
    if sentimiento == 'LABEL_4':  # Muy negativo
        return 3  # Alta urgencia
    elif sentimiento == 'LABEL_3':  # Negativo
        return 2  # Urgencia moderada
    elif sentimiento == 'LABEL_2':  # Neutro
        return 1  # Baja urgencia
    elif sentimiento == 'LABEL_1':  # Positivo
        return 1  # Baja urgencia
    elif sentimiento == 'LABEL_0':  # Muy positivo
        return 0  # Sin urgencia
    return 0  # Si no coincide con ningún sentimiento, asignar baja urgencia

# Función para verificar si el mensaje contiene palabras clave de urgencia
def urgencia_por_palabras_clave(mensaje):
    palabras_clave = ["urgente", "es urgente", "es para hoy", "necesito ayuda", "por favor", "con urgencia"]
    mensaje = mensaje.lower()
    
    for palabra in palabras_clave:
        if palabra in mensaje:
            return 1  # Incrementa urgencia
    return 0  # No hay urgencia en las palabras clave

# Función para verificar si el mensaje contiene palabras negativas
def urgencia_por_palabras_negativas(mensaje):
    palabras_negativas = ["malo", "no me gusta", "odio", "peor", "terrible", "desastroso", "fatal"]
    mensaje = mensaje.lower()
    
    for palabra in palabras_negativas:
        if palabra in mensaje:
            return 2  # Incrementa urgencia por negatividad
    return 0  # No hay urgencia en las palabras negativas

# Función para mapear las etiquetas del análisis de sentimientos a etiquetas legibles
def mapear_sentimiento(sentimiento):
    sentimiento_mapeado = {
        'LABEL_0': 'Muy Positivo',
        'LABEL_1': 'Positivo',
        'LABEL_2': 'Neutro',
        'LABEL_3': 'Negativo',
        'LABEL_4': 'Muy Negativo'
    }
    return sentimiento_mapeado.get(sentimiento, 'Desconocido')  # Si no encuentra la etiqueta, regresa "Desconocido"


# Función para calcular el nivel de urgencia
def calcular_urgencia(row):
    mensaje_filtrado = quitar_stopwords(row['Mensaje'])
    
    # Extraer bigramas y trigramas
    bigrams, trigrams = extraer_bigrams_trigrams(mensaje_filtrado)

    # Análisis de sentimiento
    result = sentiment_analysis(mensaje_filtrado)
    sentimiento = result[0]['label']
    probabilidad = result[0]['score']

    # Mapear el sentimiento a texto legible
    sentimiento_legible = mapear_sentimiento(sentimiento)

    # Si la probabilidad es baja, clasificar como "Neutro"
    if probabilidad < 0.6:
        sentimiento_legible = 'Neutro'
    
    # Clasificación de urgencia
    urgencia = urgencia_por_autor(row['Autor']) + urgencia_por_hora(row['FechaHora'].strftime('%H:%M')) + urgencia_por_sentimiento(sentimiento)
    
    urgencia += urgencia_por_palabras_clave(row['Mensaje'])
    urgencia += urgencia_por_palabras_negativas(row['Mensaje'])
    
    # Ahora, también vamos a agregar la urgencia por los bigramas y trigramas
    for bigram in bigrams + trigrams:
        bigram_str = ' '.join(bigram)
        sentiment_bigram = sentiment_analysis(bigram_str)[0]['label']
        urgencia += urgencia_por_sentimiento(sentiment_bigram)

    return min(5, urgencia), sentimiento_legible  # Regresamos la urgencia y el sentimiento legible


# Función para extraer y contar bigramas y trigramas de un DataFrame
def obtener_bigrams_trigrams(df):
    bigramas = []
    trigramas = []
    
    for mensaje in df['Mensaje']:
        bigrams, trigrams = extraer_bigrams_trigrams(mensaje)
        bigramas.extend([' '.join(bigram) for bigram in bigrams])
        trigramas.extend([' '.join(trigram) for trigram in trigrams])
    
    return bigramas, trigramas

# Función para visualizar bigramas y trigramas en un gráfico
def mostrar_grafica_bigrams_trigrams(bigramas, trigramas):
    # Contamos las ocurrencias de bigramas y trigramas
    bigram_count = Counter(bigramas).most_common(10)
    trigram_count = Counter(trigramas).most_common(10)

    # Bigramas
    bigram_df = pd.DataFrame(bigram_count, columns=['Bigram', 'Frecuencia'])
    fig_bigram = px.bar(bigram_df, x='Bigram', y='Frecuencia', title="Top 10 Bigramas más comunes")
    
    # Trigramas
    trigram_df = pd.DataFrame(trigram_count, columns=['Trigram', 'Frecuencia'])
    fig_trigram = px.bar(trigram_df, x='Trigram', y='Frecuencia', title="Top 10 Trigramas más comunes")
    
    return fig_bigram, fig_trigram

# Streamlit application code
st.title("Análisis de Chat de WhatsApp")

uploaded_file = st.file_uploader("Sube un archivo TXT de chat de WhatsApp", type=["txt"])

if uploaded_file is not None:
    df_chat = cargar_chat_txt(uploaded_file)

    if df_chat is not None and not df_chat.empty:
        st.write("Primeros mensajes del chat:")
        st.dataframe(df_chat.head())

        # Calcular la urgencia y sentimiento
        df_chat[['Urgencia', 'Sentimiento']] = df_chat.apply(calcular_urgencia, axis=1, result_type='expand')

        st.write("Mensajes con su nivel de urgencia y sentimiento:")
        st.dataframe(df_chat[['FechaHora', 'Autor', 'Mensaje', 'Urgencia', 'Sentimiento']])

        # Extraer bigramas y trigramas
        bigramas, trigramas = obtener_bigrams_trigrams(df_chat)

        # Mostrar gráficas
        fig_bigram, fig_trigram = mostrar_grafica_bigrams_trigrams(bigramas, trigramas)
        
        st.plotly_chart(fig_bigram)
        st.plotly_chart(fig_trigram)

        # Visualización del nivel de urgencia
        if st.button('Mostrar Gráfico de Urgencia'):
            urgencia_count = df_chat['Urgencia'].value_counts().reset_index()
            urgencia_count.columns = ['Urgencia', 'Cantidad']
            fig = px.bar(urgencia_count, x='Urgencia', y='Cantidad', title="Distribución de Niveles de Urgencia")
            st.plotly_chart(fig)
    else:
        st.write("No se encontraron mensajes en el archivo.")
