import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Configuração ---
st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações (NBR 14653)")

arquivo_csv = st.sidebar.file_uploader("Carregar Base de Dados (CSV)", type="csv")

if arquivo_csv is not None:
    df = pd.read_csv(arquivo_csv, sep=";", encoding='latin-1')
    features_list = ['Área Privativa', 'Área do Terreno', 'Quartos', 'Evento', 
                     'Padrão de Acabamento', 'Suite', 'Estado de Conservação', 
                     'Idade Aparente', 'Setor urbano', 'Data do Evento']
    col_alvo = 'Valor Unitário'
    
    # Limpeza
    df_clean = pd.DataFrame()
    for col in features_list + [col_alvo]:
        if col in df.columns:
            df_clean[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
    df_clean = df_clean.dropna()

    if not df_clean.empty:
        X = df_clean[features_list]
        y = df_clean[col_alvo]
        modelo = LinearRegression().fit(X, y)
        
        # 1. Equação
        eq_str = f"V.U. = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features_list, modelo.coef_)])
        st.subheader("Equação do Modelo")
        st.latex(eq_str)
        
        # 2. Diagnóstico Visual
        predicoes = modelo.predict(X)
        residuos = y - predicoes
        n, p = len(y), len(features_list) + 1
        X_mat = np.column_stack([np.ones(n), X])
        leverage = np.diag(X_mat @ np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T)
        cooks_dist = (residuos**2 / (p * np.var(residuos))) * (leverage / (1 - leverage)**2)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        ax1.scatter(y, predicoes, alpha=0.5); ax1.set_title("Aderência (Obs vs Prev)")
        ax2.scatter(predicoes, residuos, alpha=0.5, color='orange'); ax2.axhline(0, color='black', linestyle='--'); ax2.set_title("Resíduos")
        ax3.stem(cooks_dist); ax3.set_title("Distância de Cook")
        st.pyplot(fig)
        
        # 3. Inputs com Validação NBR
        st.sidebar.header("⚙️ Parâmetros do Imóvel")
        inputs = {}
        extrapolou = False
        for f in features_list:
            min_val, max_val = df_clean[f].min(), df_clean[f].max()
            val = st.sidebar.number_input(f"{f} (Limites: {min_val:.1f} a {max_val:.1f})", value=float(df_clean[f].median()))
            inputs[f] = val
            if val < min_val or val > max_val:
                st.sidebar.warning(f"⚠️ Extrapolação em {f}!")
                extrapolou = True
        
        # 4. Cálculo e Resultados
        if st.sidebar.button("Calcular Precificação"):
            if extrapolou:
                st.error("Erro: Parâmetros fora dos limites amostrais da NBR 14653.")
            else:
                pred_unit = modelo.predict(np.array([list(inputs.values())]))[0]
                erro_padrao = np.std(residuos)
                area = inputs.get('Área Privativa', 1)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("V.U. Mínimo", f"R$ {pred_unit - (1.96 * erro_padrao):,.2f}")
                col2.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
                col3.metric("V.U. Máximo", f"R$ {pred_unit + (1.96 * erro_padrao):,.2f}")
                
                st.write(f"### Valor Total Estimado: R$ {pred_unit * area:,.2f}")
                st.write(f"Intervalo Total: R$ {(pred_unit - (1.96 * erro_padrao))*area:,.2f} a R$ {(pred_unit + (1.96 * erro_padrao))*area:,.2f}")
    else:
        st.error("Dados insuficientes.")
