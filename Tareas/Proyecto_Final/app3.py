import pandas as pd
import re
from datetime import datetime
import streamlit as st
from transformers import pipeline

# Cargar el pipeline de análisis de sentimientos
#sentiment_analysis = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
sentiment_analysis = pipeline('sentiment-analysis', model='dccuchile/bert-base-spanish-wwm-uncased')


# Función para procesar el archivo .txt de WhatsApp
def cargar_chat_txt(file):
    # Leer el contenido del archivo cargado por Streamlit
    content = file.getvalue().decode('utf-8')
    
    # Dividir el contenido en líneas
    lines = content.splitlines()
    
    # Inicializar listas para almacenar los datos
    fechas = []
    autores = []
    mensajes = []

    # Patrón de expresión regular para extraer fecha, hora, autor y mensaje
    # Mejoramos el patrón para manejar los casos de hora con 'p.m.' o 'a.m.' sin puntos.
    pattern = r"(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}\s?[ap]\.?\s?[m]\.?) - (.*?):(.*)"
    
    for line in lines:
        # Ignoramos las líneas que no son relevantes como las de 'cifrado de extremo a extremo'
        if "cifrados de extremo a extremo" in line:
            continue
        
        match = re.match(pattern, line.strip())
        
        # Si hay un match, extraemos la fecha, hora, autor y mensaje
        if match:
            fecha = match.group(1)
            hora = match.group(2)
            autor = match.group(3).strip()
            mensaje = match.group(4).strip()

            # Limpiar la cadena de hora para eliminar caracteres invisibles (como \u202f) y los puntos
            hora = hora.replace("\u202f", "").strip()
            hora = hora.replace(".", "")  # Eliminar los puntos de 'p.m.' o 'a.m.'
            
            # Combinar fecha y hora en un solo campo 'FechaHora'
            fecha_hora_str = f"{fecha} {hora}"

            # Intentar convertir la cadena de fecha y hora
            try:
                # Usamos un formato de hora más flexible que permite la falta de espacio entre la hora y el AM/PM
                fecha_hora = datetime.strptime(fecha_hora_str, "%d/%m/%Y %I:%M%p")
            except ValueError as e:
                # Agregamos un mensaje de depuración si hay un error al parsear la fecha
                print(f"Error al parsear la fecha y hora: {fecha_hora_str} - {e}")
                continue

            # Agregar los datos a las listas
            fechas.append(fecha_hora)
            autores.append(autor)
            mensajes.append(mensaje)
        else:
            # Si no encontramos una coincidencia, imprimimos la línea para depuración
            print(f"No se pudo parsear la línea: {line}")
    
    # Crear un DataFrame a partir de los datos extraídos
    df = pd.DataFrame({
        'FechaHora': fechas,
        'Autor': autores,
        'Mensaje': mensajes
    })
    
    # Comprobar que las columnas necesarias están presentes
    if 'FechaHora' in df.columns and 'Autor' in df.columns and 'Mensaje' in df.columns:
        # Asegurarse de que las columnas estén en el formato correcto
        df['FechaHora'] = pd.to_datetime(df['FechaHora'])
        return df
    else:
        return None

# Función para clasificar la urgencia basada en el autor
def urgencia_por_autor(autor):
    autores_prioritarios = ["Jefe", "Hijo", "Mamá", "Papá", "Esposa"]
    return 1 if autor in autores_prioritarios else 0

# Función para clasificar la urgencia basada en la hora
def urgencia_por_hora(hora):
    hora = datetime.strptime(hora, "%H:%M")
    if hora >= datetime.strptime("20:00", "%H:%M") or hora <= datetime.strptime("05:00", "%H:%M"):
        return 1
    return 0

# Función para clasificar la urgencia basada en el sentimiento
def urgencia_por_sentimiento(sentimiento):
    # Ajustamos el análisis para considerar el sentimiento adecuadamente
    if sentimiento == 'LABEL_4':  # Muy negativo
        return 3
    elif sentimiento == 'LABEL_3':  # Negativo
        return 2
    elif sentimiento == 'LABEL_2':  # Neutro
        return 1
    elif sentimiento == 'LABEL_1':  # Positivo
        return 1
    elif sentimiento == 'LABEL_0':  # Muy positivo
        return 0
    return 0

# Función para calcular el nivel de urgencia
def calcular_urgencia(row):
    # Extraemos el sentimiento usando el pipeline
    sentiment = sentiment_analysis(row['Mensaje'])[0]['label']
    
    # Cálculo del nivel de urgencia basado en autor, hora y sentimiento
    urgencia = urgencia_por_autor(row['Autor']) + urgencia_por_hora(row['FechaHora'].strftime('%H:%M')) + urgencia_por_sentimiento(sentiment)
    
    # El nivel máximo de urgencia es 5
    return min(5, urgencia)

# Streamlit application code
# Título de la aplicación
st.title("Análisis de Chat de WhatsApp")

# Subir archivo
uploaded_file = st.file_uploader("Sube un archivo TXT de chat de WhatsApp", type=["txt"])

if uploaded_file is not None:
    # Cargar el archivo TXT usando la función 'cargar_chat_txt'
    df_chat = cargar_chat_txt(uploaded_file)

    if df_chat is not None and not df_chat.empty:
        st.write("Primeros mensajes del chat:")
        st.dataframe(df_chat.head())

        # Calcular la urgencia
        df_chat['Urgencia'] = df_chat.apply(calcular_urgencia, axis=1)

        st.write("Mensajes con su nivel de urgencia:")
        st.dataframe(df_chat[['FechaHora', 'Autor', 'Mensaje', 'Urgencia']])

        # Visualización del nivel de urgencia
        if st.button('Mostrar Gráfico de Urgencia'):
            import plotly.express as px
            urgencia_count = df_chat['Urgencia'].value_counts().reset_index()
            urgencia_count.columns = ['Urgencia', 'Cantidad']
            fig = px.bar(urgencia_count, x='Urgencia', y='Cantidad', title="Distribución de Niveles de Urgencia")
            st.plotly_chart(fig)
    else:
        st.write("No se encontraron mensajes en el archivo.")
