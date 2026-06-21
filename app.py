import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

# ... (manter as partes de leitura e limpeza de dados que já estão a funcionar)

        if not df_modelo.empty:
            X = df_modelo[features]
            y = df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)
            
            # --- DIAGNÓSTICO TÉCNICO ---
            preditos = modelo.predict(X)
            residuos = y - preditos
            
            # Cálculo da Distância de Cook (aproximada para Scikit-Learn)
            # A fórmula é baseada no resíduo e no leverage de cada ponto
            n = len(y)
            p = len(features) + 1
            leverage = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
            cooks_dist = (residuos**2 / (p * np.var(residuos))) * (leverage / (1 - leverage)**2)

            st.subheader("Diagnóstico de Diagnóstico (NBR 14653)")
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            
            # 1. Aderência
            ax[0].scatter(y, preditos, alpha=0.5, color='blue')
            ax[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            ax[0].set_title("Aderência (Obs vs Prev)")
            
            # 2. Resíduos
            ax[1].scatter(preditos, residuos, alpha=0.5, color='orange')
            ax[1].axhline(0, color='black', linestyle='--')
            ax[1].set_title("Gráfico de Resíduos")
            
            # 3. Distância de Cook
            ax[2].stem(cooks_dist)
            ax[2].set_title("Distância de Cook (Outliers)")
            
            st.pyplot(fig)
            
            # Dica visual para o Laudo
            st.info("No gráfico de Cook, barras muito acima das outras indicam imóveis 'atípicos' que merecem revisão na planilha.")
            
# ... (restante do código com o cálculo de min, médio, max)
