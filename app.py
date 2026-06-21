import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Diagnóstico Técnico")

try:
    # 1. Leitura e Limpeza (manter a lógica anterior)
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]
    
    # [Lógica de limpeza conforme usado anteriormente...]
    # ... (garantindo df_modelo com colunas numéricas)

    # 2. Regressão e Diagnóstico com Statsmodels (mais completo)
    X = df_modelo.drop(columns=[col_alvo])
    y = df_modelo[col_alvo]
    X_sm = sm.add_constant(X)
    modelo_sm = sm.OLS(y, X_sm).fit()

    # 3. Equação
    st.subheader("Equação do Modelo")
    equacao = f"V.U. = {modelo_sm.params[0]:.2f} "
    for i, col in enumerate(X.columns):
        equacao += f"+ ({modelo_sm.params[i+1]:.2f} * {col}) "
    st.latex(equacao)

    # 4. Gráficos de Diagnóstico
    st.subheader("Diagnóstico de Resíduos e Influência")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Aderência (Observado vs Previsto)
    ax[0].scatter(y, modelo_sm.fittedvalues, alpha=0.5)
    ax[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax[0].set_title("Aderência (Obs vs Prev)")
    
    # Resíduos
    ax[1].scatter(modelo_sm.fittedvalues, modelo_sm.resid, alpha=0.5, color='orange')
    ax[1].axhline(0, color='black', linestyle='--')
    ax[1].set_title("Resíduos")
    
    # Distância de Cook
    influence = modelo_sm.get_influence()
    cooks_d = influence.cooks_distance[0]
    ax[2].stem(cooks_d)
    ax[2].set_title("Distância de Cook (Outliers)")
    
    st.pyplot(fig)
    
    # 5. Valor Total e Unidade
    # Cálculo final para o imóvel inserido
    # ...

except Exception as e:
    st.error(f"Erro: {e}")
