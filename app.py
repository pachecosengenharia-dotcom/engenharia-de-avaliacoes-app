import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import io
import os
import re
import unicodedata

st.set_page_config(page_title="Engenharia de Avaliações", layout="wide")

st.title("📊 Sistema Profissional de Engenharia de Avaliações")
st.markdown("Modelagem estatística avançada e geração de laudos em conformidade com as diretrizes normativas.")

# --- GERENCIAMENTO DE BANCO DE DADOS ---
st.sidebar.header("📁 Base de Dados")

planilhas_disponiveis = []
for f in os.listdir('.'):
    if f.endswith('.csv'):
        planilhas_disponiveis.append(f)

if os.path.exists("dados"):
    for f in os.listdir("dados"):
        if f.endswith('.csv') and f not in planilhas_disponiveis:
            planilhas_disponiveis.append(os.path.join("dados", f))

fonte_dados = st.sidebar.selectbox(
    "Escolha a Região/Município",
    options=["Selecionar região salva..."] + planilhas_disponiveis + ["Fazer Upload Manual (.csv)"]
)

df = None

def carregar_csv_com_seguranca(caminho_ou_buffer):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        for sep in [';', ',']:
            try:
                if isinstance(caminho_ou_buffer, str):
                    temp_df = pd.read_csv(caminho_ou_buffer, delimiter=sep, encoding=enc)
                else:
                    caminho_ou_buffer.seek(0)
                    temp_df = pd.read_csv(caminho_ou_buffer, delimiter=sep, encoding=enc)
                
                if len(temp_df.columns) > 1:
                    return temp_df
            except:
                continue
    return None

if fonte_dados and fonte_dados != "Selecionar região salva..." and fonte_dados != "Fazer Upload Manual (.csv)":
    df = carregar_csv_com_seguranca(fonte_dados)
    if df is not None:
        st.sidebar.success(f"📍 Região carregada: {os.path.basename(fonte_dados).replace('.csv', '')}")
elif fonte_dados == "Fazer Upload Manual (.csv)":
    arquivo_upload = st.sidebar.file_uploader("Arraste sua planilha (.csv)", type=["csv"])
    if arquivo_upload:
        df = carregar_csv_com_seguranca(arquivo_upload)

# --- PROCESSAMENTO TÉCNICO DOS DADOS ---
if df is not None:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    def remover_acentos(texto):
        return ''.join(c for c in unicodedata.normalize('NFD', str(texto)) if unicodedata.category(c) != 'Mn')
    
    df.columns = [remover_acentos(col).strip() for col in df.columns]

    colunas_possiveis = {
        'Preco': ['Preco', 'Valor', 'Vlr', 'PRECO', 'Valor Total', 'Valor_Total'],
        'Area_Construida': ['Area_Construida', 'Area Construida', 'Area Const', 'Area Privativa'],
        'Area_Terreno': ['Area_Terreno', 'Area Terreno', 'Terreno', 'AREA_TERRENO', 'Area do Terreno'],
        'Quartos': ['Quartos', 'Dormitorios', 'QUARTOS'],
        'Suites': ['Suites', 'SUITES', 'Suite', 'SUITE'],
        'Vagas': ['Vagas', 'Garagem', 'VAGAS'],
        'Conservacao': ['Conservacao', 'CONSERVACAO', 'Estado de Conservacao'],
        'Padrao_Acabamento': ['Padrao_Acabamento', 'Padrao', 'Acabamento', 'PADRAO', 'Padrao de Acabamento'],
        'Setor_Urbano': ['Setor_Urbano', 'Setor Urbano', 'Setor', 'SETOR', 'SETOR_URBANO', 'Setor urbano'],
        'Data_Evento': ['Data_Evento', 'Data Evento', 'Data', 'DATA', 'Data do Evento'],
        'Evento': ['Evento', 'EVENTO']
    }

    for padrao, sinonimos in colunas_possiveis.items():
        for col in df.columns:
            if col in sinonimos:
                df = df.rename(columns={col: padrao})

    if 'Preco' not in df.columns or 'Area_Construida' not in df.columns:
        st.error(f"Não mapeamos as colunas essenciais. Cabeçalhos atuais limpos: {list(df.columns)}")
    else:
        todas_vars = ['Area_Construida', 'Area_Terreno', 'Quartos', 'Suites', 'Vagas', 'Conservacao', 'Padrao_Acabamento', 'Setor_Urbano', 'Data_Evento', 'Evento']
        variaveis_independentes = [v for v in todas_vars if v in df.columns]

        def limpar_para_numero_puro(valor):
            txt = str(valor).strip().replace('R$', '').replace(' ', '')
            if not txt or txt.lower() in ['nan', 'null', '']: return np.nan
            if ',' in txt and '.' in txt:
                if txt.find('.') < txt.find(','): txt = txt.replace('.', '')
                else: txt = txt.replace(',', '')
            if ',' in txt and '.' not in txt:
                txt = txt.replace(',', '.')
            txt = re.sub(r'[^\d.]', '', txt)
            try: return float(txt)
            except: return np.nan

        df['Preco'] = df['Preco'].apply(limpar_para_numero_puro)
        for col in variaveis_independentes:
            df[col] = df[col].apply(limpar_para_numero_puro)
            df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 1.0)

        df = df.dropna(subset=['Preco', 'Area_Construida'])

        # Painel Lateral
        st.sidebar.header("⚙️ Características do Imóvel")
        caracteristicas_avaliando = {}
        
        for var in variaveis_independentes:
            if var == 'Area_Construida':
                caracteristicas_avaliando[var] = st.sidebar.number_input("Área Construída (m²)", value=120.0, step=1.0)
            elif var == 'Area_Terreno':
                caracteristicas_avaliando[var] = st.sidebar.number_input("Área do Terreno (m²)", value=360.0, step=1.0)
            elif var in ['Quartos', 'Suites', 'Vagas']:
                caracteristicas_avaliando[var] = st.sidebar.slider(f"Quantidade de {var}", 0, 5, 2 if var != 'Quartos' else 3)
            elif var in ['Conservacao', 'Padrao_Acabamento']:
                label = "Estado de Conservação" if var == 'Conservacao' else "Padrão de Acabamento"
                caracteristicas_avaliando[var] = st.sidebar.selectbox(label, [1, 2, 3], index=1, format_func=lambda x: {1:"Baixo/Regular", 2:"Médio/Bom", 3:"Alto/Excelente"}[x])
            elif var == 'Setor_Urbano':
                valor_inicial = float(df['Setor_Urbano'].median()) if len(df) > 0 else 500.0
                caracteristicas_avaliando[var] = st.sidebar.number_input("Setor Urbano (Fator)", min_value=0.0, max_value=5000.0, value=valor_inicial, step=10.0)
            elif var == 'Data_Evento':
                valor_data = float(df['Data_Evento'].median()) if len(df) > 0 else 1.0
                caracteristicas_avaliando[var] = st.sidebar.number_input("Data do Evento (Mês/Fator)", value=valor_data, step=1.0)
            elif var == 'Evento':
                caracteristicas_avaliando[var] = st.sidebar.number_input("Fator de Evento (Venda=1.0)", value=1.0, step=0.05)

        if len(df) >= len(variaveis_independentes) + 2:
            X = df[variaveis_independentes]
            y = df['Preco']
            
            # Ajuste do Modelo
            modelo = Ridge(alpha=1.0).fit(X.values, y.values)
            
            # Predições e Diagnósticos
            y_pred_todo = modelo.predict(X.values)
            dados_imovel_lista = [caracteristicas_avaliando[var] for var in X.columns]
            preco_estimado = max(0, modelo.predict(np.array([dados_imovel_lista]))[0])
            r2_score = modelo.score(X.values, y.values)
            
            limite_inferior, limite_superior = preco_estimado * 0.85, preco_estimado * 1.15

            # Cálculo Simplificado da Distância de Cook para identificação de outliers
            residuos = y.values - y_pred_todo
            mse = np.mean(residuos ** 2)
            # Alavancagem aproximada para matrizes estáveis
            leverage = np.ones(len(df)) * (len(variaveis_independentes) / len(df))
            distancia_cook = (residuos ** 2 / (len(variaveis_independentes) * mse)) * (leverage / (1 - leverage) ** 2)
            corte_cook = 4 / len(df)

            # --- APRESENTAÇÃO NO DASHBOARD ---
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor de Mercado Estimado", f"R$ {preco_estimado:,.2f}")
            c2.metric("Intervalo Admissível (Mín/Máx)", f"R$ {limite_inferior:,.2f} a R$ {limite_superior:,.2f}")
            c3.metric("Precisão do Modelo (R²)", f"{r2_score*100:.2f}%")

            # Construção Estrita da Equação Matemática
            termos_equacao = [f"({modelo.intercept_:+,.2f})"]
            for var, coef in zip(X.columns, modelo.coef_):
                termos_equacao.append(f"({coef:+,.4f} × {var})")
            equacao_texto = "Preço = " + " ".join(termos_equacao)
            
            st.info(f"⚙️ **Equação de Regressão Gerada:** \n`{equacao_texto}`")

            # --- GERAÇÃO DOS GRÁFICOS (MATRIZ DE DIAGNÓSTICO) ---
            fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

            # 1. Dispersão Tradicional (Área vs Preço)
            sns.scatterplot(data=df, x='Area_Construida', y='Preco', color='#002d62', alpha=0.5, ax=axs[0,0], label="Amostras")
            axs[0,0].scatter([caracteristicas_avaliando['Area_Construida']], [preco_estimado], color='#d9534f', s=120, marker='*', label="Avaliando")
            axs[0,0].set_title("Dispersão: Preço vs Área", fontsize=10, weight='bold', color='#002d62')
            axs[0,0].legend(fontsize=8)

            # 2. Gráfico de Aderência (Real vs Estimado)
            axs[0,1].scatter(y.values, y_pred_todo, color='#002d62', alpha=0.5)
            axs[0,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5, label="Aderência Ideal")
            axs[0,1].set_title("Gráfico de Aderência (Real vs Estimado)", fontsize=10, weight='bold', color='#002d62')
            axs[0,1].set_xlabel("Preço Real")
            axs[0,1].set_ylabel("Preço Estimado")
            axs[0,1].legend(fontsize=8)

            # 3. Distância de Cook
            axs[1,0].stem(np.arange(len(distancia_cook)), distancia_cook, markerfmt=' ', linefmt='#002d62')
            axs[1,0].axhline(corte_cook, color='#d9534f', linestyle='--', lw=1.5, label="Limite (4/n)")
            axs[1,0].set_title("Distância de Cook (Identificação de Outliers)", fontsize=10, weight='bold', color='#002d62')
            axs[1,0].legend(fontsize=8)

            # 4. Histograma dos Resíduos
            sns.histplot(residuos, kde=True, color='#002d62', ax=axs[1,1], alpha=0.4)
            axs[1,1].set_title("Distribuição dos Resíduos (Normalidade)", fontsize=10, weight='bold', color='#002d62')

            plt.tight_layout()
            st.pyplot(fig)

            # --- ESTRUTURAÇÃO DO RELATÓRIO PDF PROFISSIONAL ---
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', dpi=250)
            img_buf.seek(0)
            
            pdf_buf = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buf, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            
            # Estilização Minimalista Luxury para o Engenheiro Consultor
            style_titulo = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#002d62'), spaceAfter=15, alignment=1)
            style_secao = ParagraphStyle('SecaoStyle', parent=styles['Heading2'], fontSize=11, textColor=colors.HexColor('#002d62'), spaceBefore=12, spaceAfter=6, borderPadding=2)
            style_texto = ParagraphStyle('TextoStyle', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor('#333333'))
            style_code = ParagraphStyle('CodeStyle', parent=styles['Normal'], fontSize=7.5, leading=10, textColor=colors.HexColor('#555555'), fontName='Courier')

            # Montagem da Tabela de Características
            dados_tabela = [["Variável Independente", "Valor Configurado para o Imóvel Avaliando"]]
            for k, v in caracteristicas_avaliando.items():
                dados_tabela.append([str(k), f"{v:,.2f}" if isinstance(v, (int, float)) else str(v)])
            
            tabela_carac = Table(dados_tabela, colWidths=[200, 250])
            tabela_carac.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (1,0), colors.HexColor('#002d62')),
                ('TEXTCOLOR', (0,0), (1,0), colors.white),
                ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('ALIGN', (1,0), (1,-1), 'RIGHT'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9f9f9')])
            ]))

            story = [
                Paragraph("LAUDO DE AVALIAÇÃO TÉCNICA MERCADOLÓGICA", style_titulo),
                Paragraph("<b>Engenharia Olfativa & Avaliações Estruturais</b>", ParagraphStyle('Sub', parent=style_texto, alignment=1)),
                Spacer(1, 15),
                
                Paragraph("1. DIAGNÓSTICO DO IMÓVEL AVALIANDO", style_secao),
                Paragraph("Abaixo estão tabeladas as características intrínsecas e extrínsecas definidas para o cálculo do modelo de regressão linear:", style_texto),
                Spacer(1, 5),
                tabela_carac,
                Spacer(1, 12),
                
                Paragraph("2. RESULTADOS DO MODELO MULTIFATORIAL", style_secao),
                Paragraph(f"<b>Valor de Mercado Estimado:</b> R$ {preco_estimado:,.2f}", style_texto),
                Paragraph(f"<b>Limite Inferior Admissível (85%):</b> R$ {limite_inferior:,.2f}", style_texto),
                Paragraph(f"<b>Limite Superior Admissível (115%):</b> R$ {limite_superior:,.2f}", style_texto),
                Paragraph(f"<b>Grau de Ajuste Estatístico (R²):</b> {r2_score*100:.2f}%", style_texto),
                Spacer(1, 8),
                
                Paragraph("3. EQUAÇÃO MATEMÁTICA DA REGRESSÃO", style_secao),
                Paragraph("A equação abaixo determina a formação de preço obtida pelo treinamento das amostras reais colhidas:", style_texto),
                Spacer(1, 4),
                Paragraph(f"{equacao_texto}", style_code),
                Spacer(1, 12),
                
                Paragraph("4. MATRIZ DE DIAGNÓSTICOS GRÁFICOS (ANEXO NBR 14653)", style_secao),
                Image(img_buf, width=480, height=320),
                Spacer(1, 10),
                Paragraph("<font size=7 color='#777777'>Relatório emitido automaticamente via plataforma de Engenharia de Avaliações.</font>", style_texto)
            ]
            
            doc.build(story)
            pdf_buf.seek(0)

            st.sidebar.markdown("---")
            st.sidebar.download_button(label="📥 Baixar Laudo Avançado (PDF)", data=pdf_buf, file_name="Laudo_Tecnico_Avancado.pdf", mime="application/pdf")
        else:
            st.warning(f"⚠️ Amostras insuficientes para gerar a matriz de diagnóstico. Linhas válidas: {len(df)}.")
else:
    st.info("💡 Escolha uma região salva no menu lateral ou selecione a opção de upload manual para começar.")
