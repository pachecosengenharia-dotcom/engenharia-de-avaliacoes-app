import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import io

# [Lógica de processamento e Regressão mantida do passo anterior]
# ... (após calcular pred_unit, pred_total, minimo, maximo e modelo)

# Gerador de PDF
def gerar_pdf(pred_unit, pred_total, minimo, maximo, equacao):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Laudo Técnico de Avaliação Imobiliária")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 770, f"Valor Unitário Estimado: R$ {pred_unit:,.2f} / m²")
    c.drawString(100, 755, f"Intervalo de Confiança (95%): R$ {minimo:,.2f} a R$ {maximo:,.2f}")
    c.drawString(100, 740, f"Valor Total Estimado: R$ {pred_total:,.2f}")
    
    c.drawString(100, 710, "Equação da Regressão:")
    c.drawString(100, 695, equacao)
    
    c.save()
    buffer.seek(0)
    return buffer

# Botão de download
if st.button("📥 Gerar Laudo PDF"):
    pdf = gerar_pdf(pred_unit, pred_total, minimo, maximo, equacao)
    st.download_button("Baixar PDF", data=pdf, file_name="laudo_tecnico.pdf")

# Gráficos
st.subheader("Análise de Aderência")
fig, ax = plt.subplots()
ax.scatter(y, modelo.predict(X), alpha=0.5)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Valores Reais")
ax.set_ylabel("Valores Estimados")
st.pyplot(fig)
