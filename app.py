import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 Engenharia de Avaliações - Debug Final")

try:
    # 1. Leitura com tratamento de separador
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]
    
    st.write("Total de linhas originais:", len(df))
    
    # 2. Limpeza Manual dos dados (substituir vírgula por ponto e remover caracteres)
    # Selecionamos todas as colunas que não são texto puro
    for col in df.columns:
        if df[col].dtype == 'object':
            # Removemos R$, espaços, e tratamos pontos/vírgulas
            df[col] = df[col].astype(str).str.replace('R$', '', regex=False)
            df[col] = df[col].astype(str).str.replace('.', '', regex=False)
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Diagnóstico crucial: quantas linhas sobram após a limpeza?
    df_limpo = df.dropna()
    st.write("Linhas após limpeza (dropna):", len(df_limpo))
    
    if len(df_limpo) == 0:
        st.error("O modelo não encontrou dados válidos. Provavelmente os números no seu CSV não estão a ser convertidos.")
        st.write("Amostra do que sobrou antes do dropna:", df.head())
    else:
        # 4. Regressão (apenas se houver dados)
        target = 'Valor Total'
        features = [c for c in df_limpo.columns if c != target and c != 'Index'] # Ajuste o nome da coluna alvo se necessário
        
        X = df_limpo[features]
        y = df_limpo[target]
        modelo = LinearRegression().fit(X, y)
        
        st.success("Modelo treinado com sucesso!")
        st.write("Variáveis usadas:", features)

except Exception as e:
    st.error(f"Erro: {e}")
