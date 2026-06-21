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

def gerar_pdf(regiao, pred_unit, minimo, maximo, eq_str, features, n_amostras):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "LAUDO TECNICO DE AVALIACAO")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Regiao: {regiao} | Amostras: {n_amostras}")
    c.drawString(50, 740, f"Equacao: {eq_str}")
    c.drawString(50, 710, f"V.U. Medio: R$ {pred_unit:,.2f}")
    c.drawString(50, 695, f"Intervalo: R$ {minimo:,.2f} a R$ {maximo:,.2f}")
    c.save()
    buffer.seek(0)
    return buffer

if regiao:
    try:
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = df.columns.str.strip()
        
        col_alvo = "Valor Unitario"
        if col_alvo in df.columns:
            excluir = ['Idade Aparente', 'Endereco', 'Complemento', 'Bairro', 'Informante', 'Telefone', 'Data do Evento']
            features = [c for c in df.columns if c != col_alvo and c not in excluir]
            if 'Setor urbano' not in features: features.append('Setor urbano')
            
            df_modelo = df[features + [col_alvo]].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.','').str.replace(',','.'), errors='coerce'))
            df_modelo = df_modelo.dropna()

            if not df_modelo.empty:
                X, y = df_modelo[features], df_modelo[col_alvo]
                modelo = LinearRegression().fit(X, y)
                
                eq_str = f"VU = {modelo.intercept_:.2f}"
                st.latex(eq_str)
                
                st.sidebar.header("Parametros")
                inputs = [st.sidebar.number_input(f"{n}", value=float(df_modelo[n].median())) for n in features]
                pred_unit = modelo.predict([inputs])[0]
                
                minimo, maximo = pred_unit * 0.90, pred_unit * 1.10
                
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Minimo", f"R$ {minimo:,.2f}")
                c2.metric("V.U. Medio", f"R$ {pred_unit:,.2f}")
                c3.metric("V.U. Maximo", f"R$ {maximo:,.2f}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.scatter(y, modelo.predict(X), alpha=0.5)
                ax1.set_title("Aderencia")
                ax2.scatter(modelo.predict(X), y - modelo.predict(X), alpha=0.5, color='orange')
                ax2.axhline(0, color='black', linestyle='--')
                ax2.set_title("Residuos")
                st.pyplot(fig)
                
                pdf_data = gerar_pdf(regiao, pred_unit, minimo, maximo, eq_str, features, len(df_modelo))
                st.download_button("Baixar Laudo PDF", data=pdf_data, file_name="laudo.pdf", mime="application/pdf")
            else:
                st.error("Dados insuficientes.")
        else:
            st.error(f"Coluna {col_alvo} nao encontrada.")
    except Exception as e:
        st.error(f"Erro no processamento: {e}")
