import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Laudo Técnico")

arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

def gerar_pdf(regiao, pred_unit, pred_total, minimo, maximo, eq_str, features, n_amostras):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "LAUDO TÉCNICO DE AVALIAÇÃO IMOBILIÁRIA")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Região: {regiao} | Amostras: {n_amostras}")
    c.drawString(50, 740, f"Equação: {eq_str}")
    c.drawString(50, 700, f"V.U. Médio: R$ {pred_unit:,.2f} | Total: R$ {pred_total:,.2f}")
    c.drawString(50, 670, f"Amplitude (95%): R$ {minimo:,.2f} a R$ {maximo:,.2f}")
    c.drawString(50, 640, "Variáveis: " + ", ".join(features))
    c.save()
    buffer.seek(0)
    return buffer

if regiao:
    try:
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = df.columns.str.strip()

        # O nome exato que apareceu no seu debug
        col_alvo = "Valor Unitário"
        
        # Filtra colunas: remove Idade Aparente, mantém Setor urbano, usa o resto
        excluir = ['Idade Aparente', 'Endereço', 'Complemento', 'Bairro', 'Informante', 'Telefone', 'Data do Evento']
        features = [c for c in df.columns if c != col_alvo and c not in excluir]
        
        # Prepara dados
        df_modelo = df[features + [col_alvo]].copy()
        for col in features + [col_alvo]:
            df_modelo[col] = pd.to_numeric(df_modelo[col].astype(str).str.replace('.','').str.replace(',','.'), errors='coerce')
        df_modelo = df_modelo.dropna()

        if not df_modelo.empty:
            X = df_modelo[features]
            y = df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)
            
            # Equação
            eq_str = f"VU = {modelo.intercept_:.2f}"
            st.latex(eq_str)

            # Inputs
            st.sidebar.header("⚙️ Parâmetros do Imóvel")
            inputs = [st.sidebar.number_input(f"{n}", value=float(df_modelo[n].median())) for n in features]
            pred_unit = modelo.predict([inputs])[0]
            
            # Resultados
            area = inputs[features.index('Área Privativa')] if 'Área Privativa' in features else 1
            minimo, maximo = pred_unit * 0.9, pred_unit * 1.1
            
            c1, c2, c3 = st.columns(3)
            c1.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
            c2.metric("Valor Total", f"R$ {pred_unit * area:,.2f}")
            
            # PDF
            pdf_data = gerar_pdf(regiao, pred_unit, pred_unit*area, minimo, maximo, eq_str, features, len(df_modelo))
            st.download_button("📥 Baixar Laudo Completo", data=pdf_data, file_name="laudo.pdf", mime="application/pdf")
        else:
            st.error("Não há dados numéricos suficientes após a limpeza.")
    except Exception as e:
        st.error(f"Erro: {e}")
