import streamlit as st
import pandas as pd
from transformers import pipeline

# Cargar el pipeline de análisis de sentimientos
sentiment_analysis = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

# Función para procesar el archivo CSV
def cargar_chat_csv(file):
    # Leer el archivo CSV
    df = pd.read_csv(file)
    
    # Asegúrate de que las columnas necesarias estén presentes
    if 'Mensaje' in df.columns and 'Fecha' in df.columns and 'Hora' in df.columns and 'Autor' in df.columns:
        # Crear una columna combinada de 'FechaHora' y convertir el formato de fecha
        # Ajustamos el formato de la fecha para considerar años con dos dígitos
        df['FechaHora'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'], format='%d/%m/%y %H:%M')
        
        # Retornar el DataFrame con las columnas relevantes
        df_chat = df[['FechaHora', 'Autor', 'Mensaje', 'Urgente', 'Grupo']]
        return df_chat
    else:
        st.error("El archivo CSV no tiene las columnas necesarias: 'Mensaje', 'Fecha', 'Hora', 'Autor'.")
        return None

# Título de la aplicación
st.title("Análisis de Chat de WhatsApp")

# Subir archivo
uploaded_file = st.file_uploader("Sube un archivo CSV de chat de WhatsApp", type=["csv"])

# Si el archivo es cargado, procesarlo
if uploaded_file is not None:
    # Cargar el archivo CSV usando la función 'cargar_chat_csv'
    df_chat = cargar_chat_csv(uploaded_file)

    if df_chat is not None:
        # Mostrar los primeros mensajes
        st.write("Primeros mensajes del chat:")
        st.dataframe(df_chat.head())

        # Realizar análisis de sentimientos y agregar la columna "Sentimiento"
        if st.button('Analizar Sentimientos'):
            # Aplicar el análisis de sentimientos a la columna 'Mensaje'
            df_chat['Sentimiento'] = df_chat['Mensaje'].apply(lambda x: sentiment_analysis(x)[0]['label'])
            st.write("Análisis de Sentimientos:")
            st.dataframe(df_chat[['FechaHora', 'Autor', 'Mensaje', 'Sentimiento']])

        # Verificar si la columna 'Sentimiento' existe antes de intentar crear el gráfico
        if 'Sentimiento' in df_chat.columns and st.button('Mostrar Gráfico de Sentimientos'):
            import plotly.express as px
            # Contar los valores de la columna 'Sentimiento'
            sentiment_count = df_chat['Sentimiento'].value_counts().reset_index()
            sentiment_count.columns = ['Sentimiento', 'Cantidad']
            # Crear el gráfico de barras
            fig = px.bar(sentiment_count, x='Sentimiento', y='Cantidad', title="Distribución de Sentimientos")
            st.plotly_chart(fig)
        elif 'Sentimiento' not in df_chat.columns:
            st.warning("Primero, haz clic en 'Analizar Sentimientos' para generar la columna de sentimientos.")
