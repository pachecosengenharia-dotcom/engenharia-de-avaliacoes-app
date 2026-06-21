# 3. Gráficos (Aderência, Resíduos e Distância de Cook)
            st.subheader("Diagnóstico Estatístico (NBR 14653)")
            
            # Cálculo da Distância de Cook
            n = len(y)
            p = len(features) + 1
            # Cálculo do leverage (matriz de projeção)
            X_mat = np.column_stack([np.ones(n), X])
            leverage = np.diag(X_mat @ np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T)
            cooks_dist = (residuos**2 / (p * np.var(residuos))) * (leverage / (1 - leverage)**2)
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            # Aderência
            ax1.scatter(y, modelo.predict(X), alpha=0.5)
            ax1.set_title("Observado vs Previsto")
            
            # Resíduos
            ax2.scatter(modelo.predict(X), residuos, alpha=0.5, color='orange')
            ax2.axhline(0, color='black', linestyle='--')
            ax2.set_title("Resíduos")
            
            # Distância de Cook
            ax3.stem(cooks_dist)
            ax3.set_title("Distância de Cook (Influência)")
            
            st.pyplot(fig)
            st.info("💡 Dica técnica: Barras muito altas no gráfico de Cook indicam imóveis atípicos na sua amostra.")
