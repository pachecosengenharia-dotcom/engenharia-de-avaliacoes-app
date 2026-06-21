import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

# Listagem de planilhas na pasta
arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        # 1. Leitura
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        # 2. Definição do alvo e das variáveis
        col_alvo = 'Valor Unitário'
        features = ['Área Construída', 'Área do Terreno', 'Evento', 'Padrão de Acabamento', 
                    'Estado de Conservação', 'Setor urbano', 'Data do Evento', 'Quartos', 'Suítes']
        
        # 3. Limpeza Seletiva
        df_modelo = pd.DataFrame()
        if col_alvo in df.columns:
            df_modelo[col_alvo] = pd.to_numeric(df[col_alvo].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
        
        for col in features:
            if col in df.columns:
                df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
        
        df_modelo = df_modelo.dropna()

        # 4. Regressão
        if not df_modelo.empty:
            X = df_modelo.drop(columns=[col_alvo])
            y = df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)
            
            # 5. Interface
            st.sidebar.header("⚙️ Parâmetros do Imóvel")
            inputs = {}
            for col in X.columns:
                inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median()))
            
            pred = modelo.predict([list(inputs.values())])
            st.metric("Valor Unitário Estimado", f"R$ {pred[0]:,.2f} / m²")
            
            st.success("Modelo treinado com sucesso!")
        else:
            st.warning("A planilha não contém dados numéricos suficientes.")

    except Exception as e:
        st.error(f"Erro: {e}")
