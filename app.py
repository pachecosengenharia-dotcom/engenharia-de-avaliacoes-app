import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

try:
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 1. FILTRAGEM DE COLUNAS: Ignorar o que não é útil
    # Removemos colunas que claramente não são numéricas ou relevantes
    colunas_para_ignorar = ['Telefone', 'Nome', 'Endereço', 'Link', 'Data']
    df_util = df.drop(columns=[c for c in colunas_para_ignorar if c in df.columns])

    # 2. Conversão segura para numérico
    def tratar_numeros(col):
        # Transforma em string, limpa formato moeda, converte para float
        return pd.to_numeric(col.astype(str).str.replace('R$', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')

    for col in df_util.columns:
        df_util[col] = tratar_numeros(df_util[col])
    
    df_util = df_util.dropna()

    # 3. Definição do Alvo (Valor Total)
    # Se o CSV tiver 'Valor Unitário', o código agora ignora-o porque o alvo é 'Valor Total'
    col_alvo = 'Valor Total'
    features = [c for c in df_util.columns if c != col_alvo]

    # 4. Regressão
    X = df_util[features]
    y = df_util[col_alvo]
    modelo = LinearRegression().fit(X, y)

    # 5. UI Limpa
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = {}
    for col in features:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_util[col].median()))

    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Estimado", f"R$ {pred[0]:,.2f}")

    # 6. Visualização Limpa (Apenas variáveis importantes)
    st.subheader("📈 Diagnóstico do Modelo")
    y_pred = modelo.predict(X)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(y, y_pred, alpha=0.5, color='darkblue')
    ax[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax[0].set_title("Aderência do Modelo")
    
    sns.histplot(y - y_pred, kde=True, ax=ax[1], color='darkgreen')
    ax[1].set_title("Distribuição dos Erros")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Erro no processamento: {e}")
