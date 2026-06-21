import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Engenharia Olfativa e Imobiliária", layout="wide")
st.title("📊 Engenharia de Avaliações - Modelo Robusto")

# --- LIMPEZA DE DADOS OTIMIZADA ---
def limpar_dados(df):
    """Remove caracteres de texto e converte tudo para float de forma segura."""
    df_novo = pd.DataFrame()
    for col in df.columns:
        # Tenta converter para numérico: remove R$, pontos, vírgulas
        s = df[col].astype(str).str.replace(r'[R$\s\.]', '', regex=True).str.replace(',', '.', regex=True)
        col_num = pd.to_numeric(s, errors='coerce')
        # Só mantemos se a coluna tiver números válidos (ignora endereços/textos)
        if col_num.notna().sum() > (len(df) * 0.5): 
            df_novo[col] = col_num
    return df_novo

try:
    # Leitura
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]
    
    # Processamento
    df_numerico = limpar_dados(df)
    df_limpo = df_numerico.dropna()
    
    st.write(f"Dados processados com sucesso! Linhas: {len(df_limpo)} | Colunas: {len(df_limpo.columns)}")
    
    # Seleção de Alvo e Features
    # O alvo será a última coluna numérica, o resto são variáveis independentes
    target = df_limpo.columns[-1]
    features = list(df_limpo.columns[:-1])
    
    # Modelo
    X = df_limpo[features]
    y = df_limpo[target]
    modelo = LinearRegression().fit(X, y)
    
    # Interface
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Parâmetros")
        inputs = {}
        for col in features:
            inputs[col] = st.number_input(f"{col}", value=float(df_limpo[col].median()))
            
    with col2:
        pred = modelo.predict([list(inputs.values())])
        st.metric("Valor Estimado", f"R$ {pred[0]:,.2f}")
        
        # Diagnóstico Visual
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.regplot(x=modelo.predict(X), y=y, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        ax.set_title("Aderência do Modelo")
        st.pyplot(fig)

except Exception as e:
    st.error(f"Erro na reanálise: {e}")
