import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
import matplotlib.pyplot as plt

# [Configuração do App e Leitura de dados conforme anterior...]
# ... (manter a leitura e limpeza de dados que já funciona)

# 1. Obter Equação da Regressão
coefs = dict(zip(X.columns, modelo.coef_))
equacao = f"V.U. = {modelo.intercept_:.2f} "
for var, coef in coefs.items():
    equacao += f"+ ({coef:.2f} * {var}) "

st.subheader("Equação da Regressão")
st.latex(equacao)

# 2. Gerar PDF
def gerar_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.drawString(100, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.drawString(100, 780, f"Equação: {equacao}")
    
    # Adicionar lista de dados ao PDF
    text = c.beginText(100, 750)
    text.textLines(df_modelo.head(10).to_string()) # Primeiros 10 dados
    c.drawText(text)
    
    c.save()
    buffer.seek(0)
    return buffer

# 3. Botão de Download do PDF
pdf_data = gerar_pdf()
st.download_button("📥 Baixar Laudo em PDF", data=pdf_data, file_name="laudo_avaliacao.pdf")

# 4. Exibir Dados e Gráficos no App
st.subheader("Lista de Dados Utilizados")
st.dataframe(df_modelo)

st.subheader("Diagnóstico de Aderência")
