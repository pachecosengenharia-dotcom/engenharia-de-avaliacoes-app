import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

try:
    # 1. Leitura e Limpeza (como já validado)
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # ... [Manter a lógica de limpeza que já funciona no seu código anterior] ...
    # (Certifique-se de que df_modelo está pronto aqui)
    
    # 2. Regressão e Cálculo de Erro
    X = df_modelo.drop(columns=[alvo_unit])
    y = df_modelo[alvo_unit]
    modelo = LinearRegression().fit(X, y)
    
    # Calcular o Erro Padrão (necessário para o intervalo de confiança)
    y_pred = modelo.predict(X)
    residuos = y - y_pred
    erro_padrao = np.std(residuos)
    
    # 3. Interface e Parâmetros
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = [st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median())) for col in X.columns]
    
    # 4. Cálculo do Ponto, Mínimo e Máximo
    pred_unit = modelo.predict([inputs])[0]
    # Intervalo de confiança (aprox. 95% de probabilidade usando 1.96 * erro)
    minimo = pred_unit - (1.96 * erro_padrao)
    maximo = pred_unit + (1.96 * erro_padrao)
    
    # Valor Total
    area_ref = inputs[0] # Assumindo que a área é a primeira feature
    
    # 5. Exibição Profissional
    st.subheader("Resultado da Avaliação")
    col1, col2, col3 = st.columns(3)
    col1.metric("Valor Unitário (Mínimo)", f"R$ {minimo:,.2f} / m²")
    col2.metric("Valor Unitário Médio", f"R$ {pred_unit:,.2f} / m²")
    col3.metric("Valor Unitário (Máximo)", f"R$ {maximo:,.2f} / m²")
    
    st.divider()
    st.metric("Valor Total Estimado (Médio)", f"R$ {pred_unit * area_ref:,.2f}")

except Exception as e:
    st.error(f"Erro: {e}")
