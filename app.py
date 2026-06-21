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

    # Mapeamento Inteligente e Ampliado de Colunas
    mapeamento_colunas = {}
    
    for col in df.columns:
        col_normalizada = col.lower().strip()
        if any(x in col_normalizada for x in ['prec', 'val', 'vlr', 'montante']): 
            mapeamento_colunas[col] = 'Preco'
        elif any(x in col_normalizada for x in ['área const', 'area const', 'área priv', 'area priv', 'área útil', 'area util']) or col_normalizada == 'area' or col_normalizada == 'área': 
            mapeamento_colunas[col] = 'Area_Construida'
        elif any(x in col_normalizada for x in ['terren', 'glen', 'área tot', 'area tot']): 
            mapeamento_colunas[col] = 'Area_Terreno'
        elif any(x in col_normalizada for x in ['quart', 'dorm', 'quar']): 
            mapeamento_colunas[col] = 'Quartos'
        elif any(x in col_normalizada for x in ['suít', 'suit']): 
            mapeamento_colunas[col] = 'Suites'
        elif any(x in col_normalizada for x in ['vag', 'garag', 'estac']): 
            mapeamento_colunas[col] = 'Vagas'
        elif any(x in col_normalizada for x in ['cons', 'estad', 'vici']): 
            mapeamento_colunas[col] = 'Conservacao'
        elif any(x in col_normalizada for x in ['acab', 'padr', 'lux']): 
            mapeamento_colunas[col] = 'Padrao_Acabamento'
        elif any(x in col_normalizada for x in ['setor', 'bairro', 'local', 'fator', 'regia', 'zona']): 
            mapeamento_colunas[col] = 'Setor_Urbano'
        elif any(x in col_normalizada for x in ['dat', 'mes', 'ano', 'epoca']): 
            mapeamento_colunas[col] = 'Data_Evento'
        elif any(x in col_normalizada for x in ['even', 'tipo_ev', 'fator_ev', 'oferta', 'venda']): 
            mapeamento_colunas[col] = 'Evento'

    df = df.rename(columns=mapeamento_colunas)
    df = df.loc[:, ~df.columns.duplicated()]

    # Obrigatoriedades mínimas estruturais
    colunas_obrigatorias = ['Preco', 'Area_Construida']
    
    if not all(c in df.columns for c in colunas_obrigatorias):
        st.error(f"Não mapeamos as colunas essenciais (Preço e Área Construída). Colunas na planilha: {list(df.columns)}")
    else:
        # Identificar dinamicamente quais das novas variáveis existem na planilha
        todas_variaveis = ['Area_Construida', 'Area_Terreno', 'Quartos', 'Suites', 'Vagas', 'Conservacao', 'Padrao_Acabamento', 'Setor_Urbano', 'Data_Evento', 'Evento']
        variaveis_independentes = [v for v in todas_variaveis if v in df.columns]

        def limpar_numero(valor):
            txt = str(valor).strip().replace('R$', '').replace(' ', '')
            if not txt or txt.lower() in ['nan', 'null', '']: return np.nan
            if ',' in txt and '.' in txt: txt = txt.replace('.', '')
            txt = txt.replace(',', '.')
            txt = re.sub(r'[^\d.]', '', txt)
            try: return float(txt)
            except: return np.nan

        # Limpeza rigorosa dos dados numéricos
        colunas_para_limpar = ['Preco'] + variaveis_independentes
        for col in colunas_para_limpar:
            df[col] = df[col].astype(str).apply(limpar_numero)
            df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 1.0)

        df = df.dropna(subset=['Preco', 'Area_Construida'])

        # Painel de Controle Lateral Dinâmico
        st.sidebar.header("⚙️ Características do Imóvel")
        caracteristicas_avaliando = {}
        
        # Gerar inputs apenas para o que de fato existe na planilha do Engenheiro
        if 'Area_Construida' in variaveis_independentes:
            caracteristicas_avaliando['Area_Construida'] = st.sidebar.number_input("Área Construída / Privativa (m²)", value=120.0, step=1.0)
        if 'Area_Terreno' in variaveis_independentes:
            caracteristicas_avaliando['Area_Terreno'] = st.sidebar.number_input("Área do Terreno (m²)", value=360.0, step=1.0)
        if 'Quartos' in variaveis_independentes:
            caracteristicas_avaliando['Quartos'] = st.sidebar.slider("Quantidade de Quartos", 1, 5, 3)
        if 'Suites' in variaveis_independentes:
            caracteristicas_avaliando['Suites'] = st.sidebar.slider("Quantidade de Suítes", 0, 5, 1)
        if 'Vagas' in variaveis_independentes:
            caracteristicas_avaliando['Vagas'] = st.sidebar.slider("Vagas de Garagem", 0, 5, 2)
        if 'Conservacao' in variaveis_independentes:
            caracteristicas_avaliando['Conservacao'] = st.sidebar.selectbox("Estado de Conservação (Nota)", [1, 2, 3], index=1, format_func=lambda x: {1:"Regular", 2:"Bom", 3:"Excelente"}[x])
        if 'Padrao_Acabamento' in variaveis_independentes:
            caracteristicas_avaliando['Padrao_Acabamento'] = st.sidebar.selectbox("Padrão de Acabamento", [1, 2, 3], index=1, format_func=lambda x: {1:"Baixo / Econômico", 2:"Médio / Normal", 3:"Alto / Luxo"}[x])
        if 'Setor_Urbano' in variaveis_independentes:
            caracteristicas_avaliando['Setor_Urbano'] = st.sidebar.number_input("Fator de Bairro / Localização", value=1.0, step=0.1)
        if 'Data_Evento' in variaveis_independentes:
            caracteristicas_avaliando['Data_Evento'] = st.sidebar.number_input("Data do Evento (Meses ou Fator)", value=1.0, step=1.0)
        if 'Evento' in variaveis_independentes:
            caracteristicas_avaliando['Evento'] = st.sidebar.number_input("Fator de Evento (Venda=1.0 / Oferta=0.9)", value=1.0, step=0.05)

        if len(df) >= len(variaveis_independentes) + 1:
            # Processamento da Regressão Linear Múltipla
            X = df[variaveis_independentes]
            y = df['Preco']
            modelo = LinearRegression().fit(X, y)
            
            dados_imovel = np.array([[caracteristicas_avaliando[var] for var in variaveis_independentes]])
            preco_estimado = max(0, modelo.predict(dados_imovel)[0])
            r2_score = modelo.score(X, y)
            limite_inferior, limite_superior = preco_estimado * 0.85, preco_estimado * 1.15

            # Exibição dos Resultados Técnicos
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor de Mercado Estimado", f"R$ {preco_estimado:,.2f}")
            c2.metric("Intervalo Admissível (Mín/Máx)", f"R$ {limite_inferior:,.2f} a R$ {limite_superior:,.2f}")
            c3.metric("Precisão do Modelo (R²)", f"{f'{r2_score*100:.2f}%' if r2_score > 0 else 'N/A'}")

            st.info(f"📐 **Variáveis processadas no cálculo multifatorial:** {', '.join(variaveis_independentes)}")

            # Gráfico de Tendência (Área Construída x Preço)
            fig, ax = plt.subplots(figsize=(8, 3.5))
            sns.scatterplot(data=df, x='Area_Construida', y='Preco', color='#002d62', alpha=0.6, ax=ax, label="Amostras de Mercado")
            ax.scatter([caracteristicas_avaliando['Area_Construida']], [preco_estimado], color='#d9534f', s=150, marker='*', label="Imóvel Avaliando")
            ax.set_title("Gráfico de Dispersão - Engenharia de Avaliações")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Geração do Relatório PDF
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
                Paragraph
