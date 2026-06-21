import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import io

# Tenta importar o reportlab, se falhar, avisa no app em vez de dar tela branca
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    has_pdf = True
except ImportError:
    has_pdf = False

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Laudo Técnico")

arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        col_alvo = 'Valor Unitário'
        # Seleção dinâmica de colunas
        features = [c for c in df.columns if c != col_alvo and pd.to_numeric(df[c].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce').notna().sum() > len(df)*0.5]
        
        df_modelo = df[features + [col_alvo]].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce'))
        df_modelo = df_modelo.dropna()

        if not df_modelo.empty:
            X = df_modelo[features]
            y = df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)
            
            # Interface
            st.sidebar.header("⚙️ Parâmetros")
            inputs = [st.sidebar.number_input(f"{n}", value=float(df_modelo[n].median())) for n in features]
            pred_unit = modelo.predict([inputs])[0]
            
            st.metric("Valor Unitário Estimado", f"R$ {pred_unit:,.2f} / m²")
            
            if has_pdf:
                st.success("PDF pronto para gerar.")
            else:
                st.warning("Biblioteca 'reportlab' não encontrada. Verifique o requirements.txt.")
        else:
            st.error("Dados insuficientes.")
    except Exception as e:
        st.error(f"Erro ao processar: {e}")
