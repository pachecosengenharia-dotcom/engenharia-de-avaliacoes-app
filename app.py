import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
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
        elif any(x in col_normalizada for x in ['área const', 'area const', 'área priv', 'area priv', 'área útil', 'area util']) or col_normalizada in ['area', 'área', 'm2', 'm²']: 
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

    # Obrigatoriedades mínimas
    if 'Preco' not in df.columns or 'Area_Construida' not in df.columns:
        st.error(f"Não mapeamos as colunas essenciais (Preço e Área Construída). Colunas identificadas na sua planilha: {list(df.columns)}")
    else:
        # Tratamento de variáveis categóricas/texto (Padrão de Acabamento)
        if 'Padrao_Acabamento' in df.columns:
            df['Padrao_Acabamento'] = df['Padrao_Acabamento'].astype(str).str.lower()
            df['Padrao_Acabamento'] = df['Padrao_Acabamento'].map({'alto': 3, 'luxo': 3, 'medio': 2, 'médio': 2, 'normal': 2, 'baixo': 1, 'economico': 1, 'econômico': 1}).fillna(2)

        def limpar_numero(valor):
            txt = str(valor).strip().replace('R$', '').replace(' ', '')
            if not txt or txt.lower() in ['nan', 'null', '']: return np.nan
            if '/' in txt:
                partes = txt.split('/')
                try: return float(partes[-1])
                except: return np.nan
            if ',' in txt and '.' in txt: txt = txt.replace('.', '')
            txt = txt.replace(',', '.')
            txt = re.sub(r'[^\d.]', '', txt)
            try: return float(txt)
            except: return np.nan

        todas_variaveis = ['Area_Construida', 'Area_Terreno', 'Quartos', 'Suites', 'Vagas', 'Conservacao', 'Padrao_Acabamento', 'Setor_Urbano', 'Data_Evento', 'Evento']
        variaveis_independentes = [v for v in todas_variaveis if v in df.columns]

        # Limpeza e conversão de dados para números puros
        df['Preco'] = df['Preco'].astype(str).apply(limpar_numero)
        for col in variaveis_independentes:
            if col != 'Padrao_Acabamento':
                df[col] = df[col].astype(str).apply(limpar_numero)
                # Calibrar preenchimento de nulos baseado no tipo de variável
                if col == 'Setor_Urbano':
                    df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 500.0)
                else:
                    df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 1.0)

        df = df.dropna(subset=['Preco', 'Area_Construida'])

        # Painel de Controle Lateral Dinâmico
        st.sidebar.header("⚙️ Características do Imóvel")
        caracteristicas_avaliando = {}
        
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
        
        # Campo Numérico calibrado para a amplitude real (50 a 1500)
        if 'Setor_Urbano' in variaveis_independentes:
            max_planilha = float(df['Setor_Urbano'].max()) if len(df) > 0 else 1500.0
            valor_padrao = float(df['Setor_Urbano'].median()) if len(df) > 0 else 500.0
            
            caracteristicas_avaliando['Setor_Urbano'] = st.sidebar.number_input(
                "Setor_Urbano", 
                min_value=0.0,
                max_value=max_planilha * 2,
                value=valor_padrao, 
                step=10.0
            )

        if 'Data_Evento' in variaveis_independentes:
            caracteristicas_avaliando['Data_Evento'] = st.sidebar.number_input("Data do Evento (Ano ou Mês)", value=2026.0, step=1.0)
        if 'Evento' in variaveis_independentes:
            caracteristicas_avaliando['Evento'] = st.sidebar.number_input("Fator de Evento (Venda=1.0 / Oferta=0.9)", value=1.0, step=0.05)

        if len(df) >= 2:
            # Regressão Ridge para estabilizar escalas numéricas muito discrepantes
            X = df[variaveis_independentes]
            y = df['Preco']
            modelo = Ridge(alpha=1.0).fit(X, y)
            
            dados_imovel = np.array([[caracteristicas_avaliando[var] for var in variaveis_independentes]])
            preco_estimado = max(0, modelo.predict(dados_imovel)[0])
            r2_score = modelo.score(X, y)
            limite_inferior, limite_superior = preco_estimado * 0.85, preco_estimado * 1.15

            # Exibição dos Resultados
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor de Mercado Estimado", f"R$ {preco_estimado:,.2f}")
            c2.metric("Intervalo Admissível (Mín/Máx)", f"R$ {limite_inferior:,.2f} a R$ {limite_superior:,.2f}")
            c3.metric("Precisão do Modelo (R²)", f"{f'{r2_score*100:.2f}%' if r2_score > 0 else 'N/A'}")

            st.info(f"📐 **Variáveis processadas no cálculo multifatorial:** {', '.join(variaveis_independentes)}")

            # Gráfico
            fig, ax = plt.subplots(figsize=(8, 3.5))
            sns.scatterplot(data=df, x='Area_Construida', y='Preco', color='#002d62', alpha=0.6, ax=ax, label="Amostras de Mercado")
            ax.scatter([caracteristicas_avaliando['Area_Construida']], [preco_estimado], color='#d9534f', s=150, marker='*', label="Imóvel Avaliando")
            ax.set_title("Gráfico de Dispersão - Engenharia de Avaliações")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Relatório PDF
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
            st.sidebar.download_button(label="📥 Baixar Laudo Oficial (PDF)", data=pdf_buf, file_name="Laudo_Tecnico_Profissional.pdf", mime="application/pdf")
        else:
            st.warning(f"Amostras insuficientes após tratamento de dados. Linhas válidas na planilha: {len(df)}.")
else:
    st.info("💡 Por favor, faça o upload de uma planilha .csv na barra lateral para iniciar.")
