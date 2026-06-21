import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Engenharia de Avaliações", layout="wide")
st.title("📊 Engenharia de Avaliações - Modelo Completo")

# 1. Carregamento e Limpeza Robusta
try:
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # Função para limpar qualquer sujeira de formatação de valores
    def limpar_numero(val):
        s = str(val).replace('R$', '').replace('.', '').replace(',', '.')
        try: return float(s)
        except: return np.nan

    # Aplicar limpeza em todas as colunas numéricas
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(limpar_numero)
    
    df = df.dropna()

    # 2. Configuração das Variáveis
    # Identificamos automaticamente o alvo (Valor) e as demais como preditoras
    target = 'Valor Total'
    features = [c for c in df.columns if c != target]
    
    st.write("Variáveis processadas:", features)

    # 3. Modelo de Regressão Multivariável
    X = df[features]
    y = df[target]
    modelo = LinearRegression().fit(X, y)

    # 4. Interface Lateral
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = {}
    for col in features:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df[col].median()))

    # 5. Cálculo e Predição
    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Estimado", f"R$ {pred[0]:,.2f}")

    # 6. Gráficos de Diagnóstico (Matriz completa)
    st.subheader("📈 Diagnóstico do Modelo (NBR 14653)")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico 1: Aderência (Real vs Estimado)
    y_pred = modelo.predict(X)
    axs[0,0].scatter(y, y_pred, alpha=0.5, color='blue')
    axs[0,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    axs[0,0].set_title("Aderência: Real vs Estimado")
    
    # Gráfico 2: Distribuição dos Resíduos
    sns.histplot(y - y_pred, kde=True, ax=axs[0,1], color='purple')
    axs[0,1].set_title("Distribuição dos Resíduos")
    
    # Gráfico 3: Resíduos vs Estimados
    axs[1,0].scatter(y_pred, y - y_pred, color='green', alpha=0.5)
    axs[1,0].axhline(0, color='red', linestyle='--')
    axs[1,0].set_title("Homocedasticidade (Resíduos vs Previstos)")
    
    # Gráfico 4: Importância das Variáveis
    coefs = pd.DataFrame({'Variável': features, 'Importância': modelo.coef_})
    sns.barplot(x='Importância', y='Variável', data=coefs, ax=axs[1,1])
    axs[1,1].set_title("Importância das Variáveis no Modelo")

    st.pyplot(fig)
    
except Exception as e:
    st.error(f"Erro: {e}")
