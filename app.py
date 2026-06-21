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
st.title("📊 AVM - Engenharia (Multi-Plataforma)")

# Função para remover acentos e normalizar nomes de colunas
def normalizar_texto(texto):
    return unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8').lower()

def gerar_laudo_pdf(d, fig, eq_str, inputs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica", 10)
    c.drawString(50, 800, "Laudo Tecnico (NBR 14653)")
    c.drawString(50, 780, f"V.U. Medio: R$ {d['vu']:,.2f} | Total: R$ {d['total']:,.2f}")
    
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 400, width=400, height=200)
    c.save(); buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar CSV", type=["csv", "txt"])
if arquivo:
    # Tenta ler com codificação robusta
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    
    # Normaliza nomes de colunas para evitar conflitos de acentuação mobile
    df.columns = [normalizar_texto(c) for c in df.columns]
    
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
            st.sidebar.header("⚙️ Parametros")
            
            # Inputs normalizados
            inputs = {}
            for f in features:
                inputs[f] = st.sidebar.number_input(f, value=float(df_c[f].median()))
            
            if st.sidebar.button("Calcular Precificacao"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                std = np.std(df_c[target] - modelo.predict(df_c[features]))
                
                # Busca por 'area' normalizada (sem acento)
                col_area = next((c for c in features if 'area' in c), None)
                total = vu * inputs[col_area] if col_area else vu
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Minimo", f"R$ {vu - (1.96*std):,.2f}")
                c2.metric("Medio", f"R$ {vu:,.2f}")
                c3.metric("Maximo", f"R$ {vu + (1.96*std):,.2f}")
                st.metric("Valor Total Estimado", f"R$ {total:,.2f}")
