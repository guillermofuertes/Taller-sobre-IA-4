import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tiktoken

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(layout="wide")

# Ocultar UI Streamlit
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.big {font-size:60px; text-align:center; font-weight:bold;}
.medium {font-size:40px; text-align:center;}
.small {font-size:25px; text-align:center;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# MODELO
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------
# ESTADO SLIDE
# -------------------------
if "slide" not in st.session_state:
    st.session_state.slide = 0

slides = [
    "titulo",
    "ia_no_piensa",
    "ia_no_piensa_2",
    "numeros",
    "token",
    "tokens_info",
    "demo_que_ve",
    "tokenizer",
    "embedding",
    "significado",
    "datos",
    "manual",
    "reales",
    "parece",
    "visualizacion",
    "gracias"
]

# -------------------------
# CONTROLES
# -------------------------
col1, col2, col3 = st.columns([1,2,1])

with col1:
    if st.button("⬅️ Anterior"):
        st.session_state.slide = max(0, st.session_state.slide - 1)

with col3:
    if st.button("Siguiente ➡️"):
        st.session_state.slide = min(len(slides)-1, st.session_state.slide + 1)

pagina = slides[st.session_state.slide]

st.write("")  # espacio visual

# -------------------------
# SLIDES
# -------------------------

if pagina == "titulo":
    st.markdown("<div class='big'>¿Cómo piensa realmente<br>la Inteligencia Artificial?</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Guillermo Fuertes</div>", unsafe_allow_html=True)

elif pagina == "ia_no_piensa":
    st.markdown("<div class='big'>La Inteligencia Artificial no piensa</div>", unsafe_allow_html=True)

elif pagina == "ia_no_piensa_2":
    st.markdown("<div class='big'>La Inteligencia Artificial no piensa</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>(al menos, como nosotros)</div>", unsafe_allow_html=True)

elif pagina == "numeros":
    st.markdown("<div class='big'>La Inteligencia Artificial no piensa</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>(al menos, como nosotros)</div>", unsafe_allow_html=True)
    st.markdown("<div class='medium'>solo entiende números</div>", unsafe_allow_html=True)

elif pagina == "token":
    st.markdown("<div class='big'>Token</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Unidades básicas en las que la IA divide el texto</div>", unsafe_allow_html=True)

elif pagina == "tokens_info":
    st.markdown("<div class='small'>Un token ≈ 4 caracteres<br>100 tokens ≈ 75 palabras<br>Un párrafo ≈ 100 tokens</div>", unsafe_allow_html=True)

# -------------------------
# DEMO ¿QUÉ VE?
# -------------------------
elif pagina == "demo_que_ve":

    placeholder = st.empty()

    placeholder.markdown("# 🐱 gato")
    time.sleep(1.5)
    placeholder.markdown("# 🔢 [0.21, -0.34, 0.88, ...]")

# -------------------------
# TOKENIZER
# -------------------------
elif pagina == "tokenizer":

    texto = st.text_area("Texto", "El gato duerme en el sofá")

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(texto)
    decoded = [encoding.decode([t]) for t in tokens]

    html = ""
    colores = ["#FFCCCC", "#CCFFCC", "#CCCCFF", "#FFFFCC"]

    for i, tok in enumerate(decoded):
        html += f"<span style='background-color:{colores[i%4]}; padding:10px; margin:5px; border-radius:6px;'>{tok}</span>"

    st.markdown(html, unsafe_allow_html=True)

# -------------------------
# EMBEDDING EXPLICACIÓN
# -------------------------
elif pagina == "embedding":
    st.markdown("<div class='big'>Embedding</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Vectores en un espacio multidimensional</div>", unsafe_allow_html=True)

elif pagina == "significado":
    st.markdown("<div class='big'>Embedding</div>", unsafe_allow_html=True)
    st.markdown("<div class='medium'>SIGNIFICADO</div>", unsafe_allow_html=True)

# -------------------------
# DATOS EDITABLES
# -------------------------
elif pagina == "datos":

    st.markdown("<div class='big'>Datos actualizados</div>", unsafe_allow_html=True)

    d1 = st.text_input("Dato 1")
    d2 = st.text_input("Dato 2")
    d3 = st.text_input("Dato 3")

    st.markdown(f"<div class='small'>{d1}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>{d2}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>{d3}</div>", unsafe_allow_html=True)

# -------------------------
# EMBEDDINGS MANUALES
# -------------------------
elif pagina == "manual":

    st.markdown("<div class='medium'>Construyendo significado</div>", unsafe_allow_html=True)

    dimensiones = ["Animal", "Comible", "Transporte"]
    palabras = ["Gato", "Manzana", "Coche"]

    df = pd.DataFrame(0.0, index=dimensiones, columns=palabras)
    df_editado = st.data_editor(df)

    if st.button("Generar"):

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(df_editado.T)

        fig, ax = plt.subplots()

        for i, palabra in enumerate(df_editado.columns):
            x, y = reduced[i]
            ax.scatter(x, y)
            ax.text(x, y, palabra)

        st.pyplot(fig)

# -------------------------
# EMBEDDINGS REALES
# -------------------------
elif pagina == "reales":

    st.markdown("<div class='medium'>Embeddings reales</div>", unsafe_allow_html=True)

    texto = st.text_input("Palabras", "gato, perro, coche")
    words = [w.strip() for w in texto.split(",")]

    embeddings = model.encode(words)

    for word, emb in zip(words, embeddings):
        vector = [float(f"{x:.3f}") for x in emb[:8]]
        st.write(f"{word}: {vector}")

# -------------------------
# PARECE QUE PIENSA
# -------------------------
elif pagina == "parece":
    st.markdown("<div class='big'>PARECE</div>", unsafe_allow_html=True)
    st.markdown("<div class='medium'>que piensa</div>", unsafe_allow_html=True)

# -------------------------
# VISUALIZACIÓN FINAL
# -------------------------
elif pagina == "visualizacion":

    st.markdown("""
    <iframe src="https://projector.tensorflow.org/" width="100%" height="600"></iframe>
    """, unsafe_allow_html=True)

# -------------------------
# GRACIAS
# -------------------------
elif pagina == "gracias":
    st.markdown("<div class='big'>Gracias</div>", unsafe_allow_html=True)
