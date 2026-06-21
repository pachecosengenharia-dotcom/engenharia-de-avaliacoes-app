import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
import unicodedata
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

def normalizar(texto):
    return unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

def gerar_laudo_pdf(d, fig, eq_str, inputs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Tecnico Completo (NBR 14653)")
    
    c.setFont("Helvetica", 10)
    # Equação com quebra de linha manual
    c.drawString(50, 790, "Equacao do Modelo:")
    y = 775
    for i in range(0, len(eq_str), 80):
        c.drawString(50, y, eq_str[i:i+80])
        y -= 15
    
    # Variáveis e seus valores
    y -= 10
    c.drawString(50, y, "Variaveis utilizadas no modelo:")
    y -= 15
    for k, v in inputs.items():
        c.drawString(60, y, f"- {k.capitalize()}: {v:.2f}")
        y -= 15
        
    # Resultados
    y -= 20
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, f"Resultado: Min (R$ {d['min']:,.2f}) | Medio (R$ {d['vu']:,.2f}) | Max (R$ {d['max']:,.2f})")
    c.drawString(50, y-20, f"Valor Total Estimado: R$ {d['total']:,.2f}")
    
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, y-250, width=400, height=200)
    c.save(); buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar CSV", type=["csv", "txt"])
if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    df.columns = [normalizar(c) for c in df.columns]
    
    cols = df.columns.tolist()
    target = st.sidebar.selectbox("Coluna Valor Unitario:", cols)
    features = st.sidebar.multiselect("Variaveis Explicativas:", [c for c in cols if c != target])
    
    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            ax1.scatter(df_c[target], modelo.predict(df_c[features])); ax1.set_title("Aderencia")
            ax2.scatter(modelo.predict(df_c[features]), df_c[target] - modelo.predict(df_c[features])); ax2.axhline(0, color='red'); ax2.set_title("Residuos")
            st.pyplot(fig)

            st.sidebar.header("⚙️ Limites de Extrapolacao")
            inputs = {}
            for f in features:
                # Mostra o limite na tela para o usuário saber onde está
                min_val, max_val = df_c[f].min(), df_c[f].max()
                inputs[f] = st.sidebar.number_input(f"{f} (Min: {min_val:.1f} | Max: {max_val:.1f})", value=float(df_c[f].median()))
            
            if st.sidebar.button("Calcular Precificacao"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                std = np.std(df_c[target] - modelo.predict(df_c[features]))
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                col_area = next((
