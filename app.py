import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia (Multi-Plataforma)")

arquivo = st.sidebar.file_uploader("Carregar Base (Universal)", type=["csv", "txt"])

if arquivo:
    # Leitura robusta para Mobile
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    
    cols = df.columns.tolist()
    target = st.sidebar.selectbox("Coluna Valor Unitário:", cols)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in cols if c != target])
    
    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            st.latex(f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)]))
            
            st.sidebar.header("⚙️ Parâmetros")
            inputs = {f: st.sidebar.number_input(f, value=float(df_c[f].median())) for f in features}
            
            # Botão de cálculo reintegrado
            if st.sidebar.button("Calcular Precificação"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                std = np.std(df_c[target] - modelo.predict(df_c[features]))
                
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo", f"R$ {vu - (1.96*std):,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {vu + (1.96*std):,.2f}")
