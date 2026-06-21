import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import io
import os
import re

st.set_page_config(page_title="Engenharia de Avaliações", layout="wide")

st.title("📊 Sistema Profissional de Engenharia de Avaliações")
st.markdown("Modelagem estatística multifatorial para laudos técnicos em conformidade normativa.")

# --- CARREGAMENTO SIMPLIFICADO ---
st.sidebar.header("📁 Base de Dados")

planilhas_disponiveis = [f for f in os.listdir('.') if f.endswith('.csv')]
if os.path.exists("dados"):
    for f in os.listdir("dados"):
        if f.endswith('.csv') and f not in planilhas_disponiveis:
            planilhas_disponiveis.append(os.path.join("dados", f))

fonte_dados = st.sidebar.selectbox(
    "Escolha a Região/Município",
    options=["Selecionar região salva..."] + planilhas_disponiveis + ["Fazer Upload Manual (.csv)"]
)

df = None

if fonte_dados and fonte_dados != "Selecionar região salva..." and fonte_dados != "Fazer Upload Manual (.csv)":
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(fonte_dados, delimiter=';', encoding=enc)
            if len(df.columns) > 1: break
        except:
            continue

# --- PROCESSAMENTO DIRETOR ---
if df is not None:
    # Ajuste de cabeçalhos específicos da planilha
    df.columns = [col.strip() for col in df.columns]
    
    # Mapeamento estrito para Goiânia
    mapeamento = {
        'Área Privativa': 'Area_Construida',
        'Área do Terreno': 'Area_Terreno',
        'Valor Total': 'Preco',
        'Estado de Conservação': 'Conservacao',
        'Padrão de Acabamento': 'Padrao_Acabamento',
        'Setor urbano': 'Setor_Urbano',
        'Idade Aparente': 'Idade_Aparent'
    }
    df = df.rename(columns=mapeamento)

    # Forçar conversão numérica limpa
    def limpar_numero(v):
        if pd.isna(v): return np.nan
        txt = str(v).replace('R$', '').replace(' ', '')
        if ',' in txt and '.' in txt:
            if txt.find('.') < txt.find(','): txt = txt.replace('.', '')
            else: txt = txt.replace(',', '')
        elif ',' in txt:
            txt = txt.replace(',', '.')
        txt = re.sub(r'[^\d.]', '', txt)
        try: return float(txt)
        except: return np.nan

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(limpar_numero)

    # Definir variáveis para o cálculo técnico
    vars_calculo = ['Area_Construida', 'Quartos', 'Suite']
    vars_disponiveis = [v for v in vars_calculo if v in df.columns]
    
    if 'Preco' in df.columns and 'Area_Construida' in df.columns:
        df = df.dropna(subset=['Preco', 'Area_Construida'])
        
        # Preencher vazios com a mediana
        for v in vars_disponiveis:
            df[v] = df[v].fillna(df[v].median() if not df[v].isnull().all() else 1.0)

        # Painel Lateral de Entrada
        st.sidebar.header("⚙️ Características do Imóvel")
        valores_avaliando = {}
        valores_avaliando['Area_Construida'] = st.sidebar.number_input("Área Construída / Privativa (m²)", value=120.0, step=1.0)
        if 'Quartos' in vars_disponiveis:
            valores_avaliando['Quartos'] = float(st.sidebar.slider("Quantidade de Quartos", 0, 5, 3))
        if 'Suite' in vars_disponiveis:
            valores_avaliando['Suite'] = float(st.sidebar.slider("Quantidade de Suítes", 0, 5, 1))

        if len(df) >= len(vars_disponiveis) + 2:
            X = df[vars_disponiveis].values
            y = df['Preco'].values

            # Modelo matemático puro sem escalonamento para evitar conflito de objetos
            modelo_puro = LinearRegression()
            modelo_puro.fit(X, y)
            
            y_pred = modelo_puro.predict(X)
            
            # Predição do Imóvel Avaliando
            dados_imovel = np.array([valores_avaliando[v] for v in vars_disponiveis]).reshape(1, -1)
            preco_estimado = max(0, float(modelo_puro.predict(dados_imovel)[0]))
            
            # Coeficientes
            intercepto = float(modelo_puro.intercept_)
            coeficientes = modelo_puro.coef_
            
            r2_score = modelo_puro.score(X, y)
            limite_inferior, limite_superior = preco_estimado * 0.85, preco_estimado * 1.15

            # Apresentação dos Resultados Principais
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor de Mercado Estimado", f"R$ {preco_estimado:,.2f}")
            c2.metric("Intervalo Admissível (Mín/Máx)", f"R$ {limite_inferior:,.2f} a R$ {limite_superior:,.2f}")
            c3.metric("Precisão do Modelo (R²)", f"{r2_score*100:.2f}%")

            # Montagem da Equação Analítica
            equacao_texto = f"Preço = {intercepto:,.2f}"
            for var, coef in zip(vars_disponiveis, coeficientes):
                sinal = "+" if coef >= 0 else "-"
                equacao_texto += f" {sinal} {abs(coef):,.2f} × {var}"
            
            st.info(f"📐 **Equação de Regressão Linear Obtida:** \n`{equacao_texto}`")

            # --- MATRIZ DE DIAGNÓSTICOS GRÁFICOS ---
            fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
            residuos = y - y_pred

            # 1. Dispersão Real
            axs[0,0].scatter(df['Area_Construida'].values, y, color='#002d62', alpha=0.5, label="Amostras")
            axs[0,0].scatter([valores_avaliando['Area_Construida']], [preco_estimado], color='#d9534f', s=130, marker='*', zorder=5, label="Avaliando")
            axs[0,0].set_title("Dispersão: Preço vs Área", fontsize=10, weight='bold')
            axs[0,0].set_xlabel("Área Construída (m²)")
            axs[0,0].set_ylabel("Preço (R$)")
            axs[0,0].legend(fontsize=8)

            # 2. Aderência Real
            axs[0,1].scatter(y, y_pred, color='#002d62', alpha=0.5)
            axs[0,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5, label="Ideal")
            axs[0,1].set_title("Gráfico de Aderência", fontsize=10, weight='bold')
            axs[0,1].legend(fontsize=8)

            # 3. Resíduos
            axs[1,0].scatter(y_pred, residuos, color='#002d62', alpha=0.5)
            axs[1,0].axhline(0, color='red', linestyle='--')
            axs[1,0].set_title("Resíduos vs Valores Estimados", fontsize=10, weight='bold')

            # 4. Histograma
            sns.histplot(residuos, kde=True, color='#002d62', ax=axs[1,1], alpha=0.4)
            axs[1,1].set_title("Distribuição dos Erros (Normalidade)", fontsize=10, weight='bold')

            plt.tight_layout()
            st.pyplot(fig)

            # --- RELATÓRIO PDF ---
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', dpi=200)
            img_buf.seek(0)
            
            pdf_buf = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buf, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            
            style_titulo = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=15, textColor=colors.HexColor('#002d62'), spaceAfter=12, alignment=1)
            style_secao = ParagraphStyle('SecaoStyle', parent=styles['Heading2'], fontSize=11, textColor=colors.HexColor('#002d62'), spaceBefore=10, spaceAfter=5)
            style_texto = ParagraphStyle('TextoStyle', parent=styles['Normal'], fontSize=9, leading=13)

            dados_tabela = [["Variável", "Valor Configurado"]]
            for k, v in valores_avaliando.items():
                dados_tabela.append([str(k), f"{v:,.2f}" if isinstance(v, (int, float)) else str(v)])
            
            tabela_carac = Table(dados_tabela, colWidths=[200, 250])
            tabela_carac.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (1,0), colors.HexColor('#002d62')),
                ('TEXTCOLOR', (0,0), (1,0), colors.white),
                ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd'))
            ]))

            story =
