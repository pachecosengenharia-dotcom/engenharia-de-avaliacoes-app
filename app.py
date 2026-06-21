import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import io
import re

st.set_page_config(page_title="Engenharia de Avaliações", layout="wide")

st.title("📊 Sistema Profissional de Engenharia de Avaliações")
st.markdown("Insira sua planilha de mercado e ajuste os parâmetros para calcular o valor de mercado.")

# Barra Lateral - Upload dos Dados
st.sidebar.header("📁 Base de Dados")
arquivo_upload = st.sidebar.file_uploader("Arraste sua planilha (.csv)", type=["csv"])

if arquivo_upload:
    try:
        df = pd.read_csv(arquivo_upload, delimiter=';', encoding='latin-1')
        if len(df.columns) <= 1:
            df = pd.read_csv(arquivo_upload, delimiter=',', encoding='latin-1')
    except:
        df = pd.read_csv(arquivo_upload, delimiter=',', encoding='latin-1')
        
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Mapeamento Inteligente de Colunas
    mapeamento_colunas = {}
    colunas_encontradas = set()
    
    for col in df.columns:
        col_normalizada = col.lower().strip()
        if any(x in col_normalizada for x in ['prec', 'val', 'vlr', 'montante']): 
            mapeamento_colunas[col] = 'Preco'
            colunas_encontradas.add('Preco')
        elif any(x in col_normalizada for x in ['are', 'm2', 'm²', 'metrag', 'dimen']): 
            mapeamento_colunas[col] = 'Area'
            colunas_encontradas.add('Area')
        elif any(x in col_normalizada for x in ['quart', 'dorm', 'quar']): 
            mapeamento_colunas[col] = 'Quartos'
            colunas_encontradas.add('Quartos')
        elif any(x in col_normalizada for x in ['vag', 'garag', 'estac']): 
            mapeamento_colunas[col] = 'Vagas'
            colunas_encontradas.add('Vagas')
        elif any(x in col_normalizada for x in ['cons', 'estad', 'vici']): 
            mapeamento_colunas[col] = 'Conservacao'
            colunas_encontradas.add('Conservacao')
        elif any(x in col_normalizada for x in ['setor', 'bairro', 'local', 'fator', 'regia', 'zona']): 
            mapeamento_colunas[col] = 'Setor_Urbano'
            colunas_encontradas.add('Setor_Urbano')

    df = df.rename(columns=mapeamento_colunas)
    df = df.loc[:, ~df.columns.duplicated()]

    # Obrigatoriedades mínimas para rodar uma avaliação técnica
    colunas_obrigatorias = ['Preco', 'Area']
    
    if not all(c in df.columns for c in colunas_obrigatorias):
        st.error(f"Não mapeamos as colunas essenciais (Preço e Área). Colunas na planilha: {list(df.columns)}")
    else:
        # Definir quais variáveis independentes vão entrar no modelo dinamicamente
        variaveis_independentes = ['Area']
        for var in ['Quartos', 'Vagas', 'Conservacao', 'Setor_Urbano']:
            if var in df.columns:
                variaveis_independentes.append(var)

        def limpar_numero(valor):
            txt = str(valor).strip().replace('R$', '').replace(' ', '')
            if not txt or txt.lower() in ['nan', 'null', '']: return np.nan
            if ',' in txt and '.' in txt: txt = txt.replace('.', '')
            txt = txt.replace(',', '.')
            txt = re.sub(r'[^\d.]', '', txt)
            try: return float(txt)
            except: return np.nan

        # Limpar apenas as colunas que mapeamos e existem de facto
        colunas_para_limpar = ['Preco'] + variaveis_independentes
        for col in colunas_para_limpar:
            df[col] = df[col].astype(str).apply(limpar_numero)
            df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 1.0)

        df = df.dropna(subset=['Preco', 'Area'])

        # Painel de Controle - Características Dinâmicas
        st.sidebar.header("⚙️ Características do Imóvel")
        caracteristicas_avaliando = {}
        
        caracteristicas_avaliando['Area'] = st.sidebar.number_input("Área Útil (m²)", value=120.0, step=1.0)
        
        if 'Quartos' in variaveis_independentes:
            caracteristicas_avaliando['Quartos'] = st.sidebar.slider("Quantidade de Quartos", 1, 5, 3)
        if 'Vagas' in variaveis_independentes:
            caracteristicas_avaliando['Vagas'] = st.sidebar.slider("Vagas de Garagem", 0, 5, 2)
        if 'Conservacao' in variaveis_independentes:
            caracteristicas_avaliando['Conservacao'] = st.sidebar.selectbox("Estado de Conservação", [1, 2, 3], index=1, format_func=lambda x: {1:"Regular", 2:"Bom", 3:"Excelente"}[x])
        if 'Setor_Urbano' in variaveis_independentes:
            caracteristicas_avaliando['Setor_Urbano'] = st.sidebar.number_input("Fator de Bairro / Setor Urbano", value=1.0, step=0.1)

        if len(df) >= 3:
            # Construção do modelo dinâmico
            X = df[variaveis_independentes]
            y = df['Preco']
            modelo = LinearRegression().fit(X, y)
            
            # Montar o vetor do avaliando na ordem correta das colunas
            dados_imovel = np.array([[caracteristicas_avaliando[var] for var in variaveis_independentes]])
            
            preco_estimado = max(0, modelo.predict(dados_imovel)[0])
            r2_score = modelo.score(X, y)
            limite_inferior, limite_superior = preco_estimado * 0.85, preco_estimado * 1.15

            # Exibição dos Resultados
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor de Mercado Estimado", f"R$ {preco_estimado:,.2f}")
            c2.metric("Intervalo Admissível (Mín/Máx)", f"R$ {limite_inferior:,.2f} a R$ {limite_superior:,.2f}")
            c3.metric("Precisão do Modelo (R²)", f"{f'{r2_score*100:.2f}%' if r2_score > 0 else 'N/A'}")

            # Identificar dinamicamente as variáveis usadas para avisar o Engenheiro
            st.info(f"📐 Modelo calculado com base em: {', '.join(variaveis_independentes)}")

            # Gráfico
            fig, ax = plt.subplots(figsize=(8, 3.5))
            sns.scatterplot(data=df, x='Area', y='Preco', color='#002d62', alpha=0.6, ax=ax, label="Amostras")
            ax.scatter([caracteristicas_avaliando['Area']], [preco_estimado], color='#d9534f', s=150, marker='*', label="Avaliando")
            ax.set_title("Modelo de Regressão Linear Adaptativo")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', dpi=200)
            img_buf.seek(0)
            
            pdf_buf = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buf, pagesize=letter)
            styles = getSampleStyleSheet()
            
            detalhes_texto = " | ".join([f"<b>{k}:</b> {v}" for k, v in caracteristicas_avaliando.items()])
            story = [
                Paragraph("LAUDO DE AVALIAÇÃO TÉCNICA MERCADOLÓGICA", ParagraphStyle('T', fontSize=18, textColor=colors.HexColor('#002d62'), alignment=1)),
                Spacer(1, 15),
                Paragraph(detalhes_texto, styles['Normal']),
                Spacer(1, 5),
                Paragraph(f"<b>Valor de Mercado Inferido: R$ {preco_estimado:,.2f}</b>", styles['Normal']),
                Spacer(1, 15),
                Image(img_buf, width=400, height=180)
            ]
            doc.build(story)
            pdf_buf.seek(0)

            st.sidebar.markdown("---")
            st.sidebar.download_button(label="📥 Baixar Laudo Oficial (PDF)", data=pdf_buf, file_name="Laudo_Profissional.pdf", mime="application/pdf")
        else:
            st.warning("Dados insuficientes na planilha para gerar cálculos técnicos.")
else:
    st.info("💡 Por favor, faça o upload de uma planilha .csv na barra lateral para iniciar.")
