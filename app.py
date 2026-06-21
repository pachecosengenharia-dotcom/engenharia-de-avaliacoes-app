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
        modelo = LinearRegression().fit(df_clean[features_list], df_clean[col_alvo])
        
        # --- INPUTS COM VALIDAÇÃO DE LIMITES ---
        st.sidebar.header("⚙️ Parâmetros do Imóvel")
        inputs = {}
        extrapolou = False
        
        for f in features_list:
            min_val, max_val = df_clean[f].min(), df_clean[f].max()
            val = st.sidebar.number_input(f"{f} (Limites: {min_val:.1f} a {max_val:.1f})", 
                                          value=float(df_clean[f].median()))
            inputs[f] = val
            
            # Verificação de extrapolação conforme NBR 14653
            if val < min_val or val > max_val:
                st.sidebar.warning(f"⚠️ Extrapolação em {f}!")
                extrapolou = True
        
        if st.sidebar.button("Calcular Precificação"):
            if extrapolou:
                st.error("Atenção: O modelo não pode ser utilizado com segurança, pois existem variáveis fora dos limites amostrais da NBR 14653.")
            else:
                # Predição
                input_array = np.array([inputs[f] for f in features_list]).reshape(1, -1)
                pred_unit = modelo.predict(input_array)[0]
                
                # Intervalo e Total
                residuos = df_clean[col_alvo] - modelo.predict(df_clean[features_list])
                erro_padrao = np.std(residuos)
                area = inputs.get('Área Privativa', 1)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("V.U. Mínimo", f"R$ {pred_unit - (1.96 * erro_padrao):,.2f}")
                col2.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
                col3.metric("V.U. Máximo", f"R$ {pred_unit + (1.96 * erro_padrao):,.2f}")
                st.write(f"### Valor Total Estimado: R$ {pred_unit * area:,.2f}")
    else:
        st.error("Erro na leitura dos dados.")
