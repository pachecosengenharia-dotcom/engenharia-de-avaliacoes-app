import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Modelo de Valor Unitário")

try:
    # 1. Leitura
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 2. Definição explícita do alvo
    # O sistema agora foca no Valor Unitário como dependente
    col_alvo = 'Valor Unitário' 
    
    # Função de limpeza para converter tudo em float
    def limpar_num(val):
        s = str(val).replace('R$', '').replace('.', '').replace(',', '.')
        try: return float(s)
        except: return np.nan

    # 3. Processamento de dados
    df_limpo = pd.DataFrame()
    for col in df.columns:
        if col == col_alvo:
            df_limpo[col] = df[col].apply(limpar_num)
        elif df[col].dtype == 'object':
            conv = df[col].apply(limpar_num)
            if conv.notna().sum() > (len(df) * 0.7):
                df_limpo[col] = conv
        else:
            df_limpo[col] = df[col]

    df_limpo = df_limpo.dropna()

    # 4. Regressão
    X = df_limpo.drop(columns=[col_alvo])
    y = df_limpo[col_alvo]
    modelo = LinearRegression().fit(X, y)

    # 5. Interface
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = {}
    for col in X.columns:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_limpo[col].median()))

    # 6. Predição
    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Unitário Estimado", f"R$ {pred[0]:,.2f} / m²")

    # 7. Diagnóstico Visual
    st.subheader("📈 Diagnóstico de Aderência")
    y_pred = modelo.predict(X)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.regplot(x=y_pred, y=y, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    ax.set_xlabel("Estimado")
    ax.set_ylabel("Real")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Erro na reconfiguração do modelo: {e}")
