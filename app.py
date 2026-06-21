import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

st.title("📊 Engenharia de Avaliações - Modo de Segurança")

# Busca arquivos
arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        # Leitura bruta forçando formato
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = df.columns.str.strip()
        
        # Filtra apenas o que é número, ignora erros
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('.','').str.replace(',','.'), errors='coerce')
        
        df = df.dropna()
        
        if not df.empty:
            # Assume que a última coluna é o Valor Unitário
            col_alvo = df.columns[-1]
            features = [c for c in df.columns if c != col_alvo]
            
            X = df[features]
            y = df[col_alvo]
            
            modelo = LinearRegression().fit(X, y)
            
            st.write(f"Modelo treinado com {len(df)} amostras.")
            st.write(f"Variável alvo: {col_alvo}")
            
            st.sidebar.header("Parâmetros")
            inputs = [st.sidebar.number_input(f"{n}", value=float(df[n].median())) for n in features]
            
            pred = modelo.predict([inputs])[0]
            st.metric("Resultado Estimado", f"R$ {pred:,.2f}")
        else:
            st.error("O arquivo está vazio ou sem dados numéricos.")
            
    except Exception as e:
        st.error(f"Erro crítico: {e}")
