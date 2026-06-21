import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

try:
    # 1. Leitura do ficheiro com tratamento de codificação
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 2. Função "Blindada" para limpar os dados
    def tratar_coluna(col):
        # Transforma tudo em string, tira pontos de milhar, troca vírgula por ponto
        return pd.to_numeric(col.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')

    # Aplicar a limpeza em todas as colunas
    for col in df.columns:
        if col != 'Valor Total': # Mantemos o alvo como referência
             df[col] = tratar_coluna(df[col])
    
    df['Valor Total'] = tratar_coluna(df['Valor Total'])
    
    # Removemos linhas que ficaram vazias após a limpeza
    df = df.dropna()

    # 3. Definição das variáveis automaticamente
    target = 'Valor Total'
    features = [c for c in df.columns if c != target]
    
    st.write("Variáveis detectadas:", features)

    # 4. Regressão
    X = df[features]
    y = df[target]
    modelo = LinearRegression().fit(X, y)

    # 5. Interface
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = {}
    for col in features:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df[col].median()))

    # 6. Cálculo
    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Estimado", f"R$ {pred[0]:,.2f}")

    # 7. Diagnóstico Gráfico (NBR 14653)
    st.subheader("📈 Diagnóstico do Modelo")
    y_pred = modelo.predict(X)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(y, y_pred, color='blue', alpha=0.5)
    ax[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax[0].set_title("Aderência (Real vs Estimado)")
    
    sns.histplot(y - y_pred, kde=True, ax=ax[1], color='purple')
    ax[1].set_title("Distribuição dos Resíduos")
    
    st.pyplot(fig)

except Exception as e:
    st.error(f"Erro na limpeza dos dados: {e}")
