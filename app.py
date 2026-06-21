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
import os

st.set_page_config(page_title="Engenharia de Avaliações", layout="wide")

st.title("📊 Sistema Profissional de Engenharia de Avaliações")
st.markdown("Selecione uma região salva no seu repositório ou insira uma nova planilha de mercado.")

# --- GERENCIAMENTO DE BANCO DE DADOS INTELIGENTE ---
st.sidebar.header("📁 Base de Dados")

# Procura arquivos .csv na raiz do projeto e na pasta 'dados' para facilitar no celular
planilhas_disponiveis = []
for f in os.listdir('.'):
    if f.endswith('.csv'):
        planilhas_disponiveis.append(f)

if os.path.exists("dados"):
    for f in os.listdir("dados"):
        if f.endswith('.csv') and f not in planilhas_disponiveis:
            planilhas_disponiveis.append(os.path.join("dados", f))

# Menu seletor de regiões
fonte_dados = st.sidebar.selectbox(
    "Escolha a Região/Município",
    options=["Selecionar região salva..."] + planilhas_disponiveis + ["Fazer Upload Manual (.csv)"]
)

df = None

# Função robusta de leitura com detecção automática de codificação e separador
def carregar_csv_com_seguranca(caminho_ou_buffer):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        for sep in [';', ',']:
            try:
                # Se for uma string (caminho de arquivo), lê direto, senão lê o buffer do upload
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

# Carregamento baseado na escolha do usuário
if fonte_dados and fonte_dados != "Selecionar região salva..." and fonte_dados != "Fazer Upload Manual (.csv)":
    df = carregar_csv_com_seguranca(fonte_dados)
    if df is not None:
        nome_exibicao = os.path.basename(fonte_dados).replace('.csv', '')
        st.sidebar.success(f"📍 Região carregada com sucesso: {nome_exibicao}")

elif fonte_dados == "Fazer Upload Manual (.csv)":
    arquivo_upload = st.sidebar.file_uploader("Arraste sua planilha (.csv)", type=["csv"])
    if arquivo_upload:
        df = carregar_csv_com_seguranca(arquivo_upload)

# --- PROCESSAMENTO TÉCNICO DOS DADOS ---
if df is not None:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.astype(str).str.strip()

    # Dicionário calibrado EXATAMENTE com os cabeçalhos reais das suas planilhas
    colunas_possiveis = {
        'Preco': ['Preco', 'Preço', 'Valor', 'Vlr', 'PRECO', 'PREÇO', 'Valor Total', 'Valor_Total'],
        'Area_Construida': ['Area_Construida', 'Área_Construída', 'Area Construida', 'Área Construída', 'Area Const', 'Área Const', 'AREA_CONSTRUIDA', 'Área Privativa', 'Area Privativa'],
        'Area_Terreno': ['Area_Terreno', 'Área_Terreno', 'Area Terreno', 'Terreno', 'AREA_TERRENO', 'Área do Terreno', 'Area do Terreno'],
        'Quartos': ['Quartos', 'Dormitorios', 'Dormitórios', 'QUARTOS'],
        'Suites': ['Suites', 'Suítes', 'SUITES', 'Suite', 'Suíte'],
        'Vagas': ['Vagas', 'Garagem', 'VAGAS'],
        'Conservacao': ['Conservacao', 'Conservação', 'CONSERVACAO', 'Estado de Conservação', 'Estado de Conservacao'],
        'Padrao_Acabamento': ['Padrao_Acabamento', 'Padrão_Acabamento', 'Padrao', 'Padrão', 'Acabamento', 'PADRAO', 'Padrão de Acabamento', 'Padrao de Acabamento'],
        'Setor_Urbano': ['Setor_Urbano', 'Setor Urbano', 'Setor', 'SETOR', 'SETOR_URBANO', 'Setor urbano'],
        'Data_Evento': ['Data_Evento', 'Data Evento', 'Data', 'DATA', 'Data do Evento'],
        'Evento': ['Evento', 'EVENTO']
    }

    for padrao, sinonimos in colunas_possiveis.items():
        for col in df.columns:
            if col in sinonimos:
                df = df.rename(columns={col: padrao})

    if 'Preco' not in df.columns or 'Area_Construida' not in df.columns:
        st.error(f"Não mapeamos as colunas essenciais (Preço e Área). Verifique os cabeçalhos. Colunas atuais detectadas: {list(df.columns)}")
    else:
        todas_vars = ['Area_Construida', 'Area_Terreno', 'Quartos', 'Suites', 'Vagas', 'Conservacao', 'Padrao_Acabamento', 'Setor_Urbano', 'Data_Evento', 'Evento']
        variaveis_independentes = [v for v in todas_vars if v in df.columns]

        def limpar_numero(valor):
            txt = str(valor).strip().replace('R$', '').replace(' ', '')
            if not txt or txt.lower() in ['nan', 'null', '']: return np.nan
            if ',' in txt and '.' in txt:
                if txt.find('.') < txt.find(','): txt = txt.replace('.', '')
                else: txt = txt.replace(',', '')
            if ',' in txt and '.' not in txt: txt = txt.replace(',', '.')
            txt = re.sub(r'[^\d.]', '', txt)
            try: return float(txt)
            except: return np.nan

        df['Preco'] = df['Preco'].astype(str).apply(limpar_numero)
        for col in variaveis_independentes:
            df[col] = df[col].astype(str).apply(limpar_numero)
            df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 1.0)

        df = df.dropna(subset=['Preco', 'Area_Construida'])

        # Painel de Controle Lateral Dinâmico
        st.sidebar.header("⚙️ Características do Imóvel")
        caracteristicas_avaliando = {}
        
        if 'Area_Construida' in variaveis_independentes:
            caracteristicas_avaliando['Area_Construida'] = st.sidebar.number_input("Área Construída (m²)", value=120.0, step=1.0)
        if 'Area_Terreno' in variaveis_independentes:
            caracteristicas_avaliando['Area_Terreno'] = st.sidebar.number_input("Área do Terreno (m²)", value=360.0, step=1.0)
        if 'Quartos' in variaveis_independentes:
            caracteristicas_avaliando['Quartos'] = st.sidebar.slider("Quantidade de Quartos", 1, 5, 3)
        if 'Suites' in variaveis_independentes:
            caracteristicas_avaliando['Suites'] = st.sidebar.slider("Quantidade de Suítes", 0, 5, 1)
        if 'Vagas' in variaveis_independentes:
            caracteristicas_avaliando['Vagas'] = st.sidebar.slider("Vagas de Garagem", 0, 5, 2)
        if 'Conservacao' in variaveis_independentes:
            caracteristicas_avaliando['Conservacao'] = st.sidebar.selectbox("Estado de Conservação", [1, 2, 3], index=1, format_func=lambda x: {1:"Regular", 2:"Bom", 3:"Excelente"}[x])
        if 'Padrao_Acabamento' in variaveis_independentes:
            caracteristicas_avaliando['Padrao_Acabamento'] = st.sidebar.selectbox("Padrão de Acabamento", [1, 2, 3], index=1, format_func=lambda x: {1:"Baixo", 2:"Médio", 3:"Alto"}[x])
        
        if 'Setor_Urbano' in variaveis_independentes:
            valor_inicial = float(df['Setor_Urbano'].median()) if len(df) > 0 else 500.0
            caracteristicas_avaliando['Setor_Urbano'] = st.sidebar.number_input("Setor_Urbano", min_value=0.0, max_value=5000.0, value=valor_inicial, step=10.0)
        if 'Data_Evento' in variaveis_independentes:
            caracteristicas_avaliando['Data_Evento'] = st.sidebar.number_input("Data do Evento", value=2026.0, step=1.0)
        if 'Evento' in variaveis_independentes:
            caracteristicas_avaliando['Evento'] = st.sidebar.number_input("Fator de Evento", value=1.0, step=0.05)

        if len(df) >= 2:
            X = df[variaveis_independentes]
            y = df['Preco']
            
            modelo = Ridge(alpha=1.0).fit(X.values, y.values)
            
            dados_imovel_lista = [caracteristicas_avaliando[var] for var in variaveis_independentes]
            dados_imovel = np.array([dados_imovel_lista])
            
            preco_estimado = max(0, modelo.predict(dados_imovel)[0])
            r2_score = modelo.score(X.values, y.values)
            limite_inferior, limite_superior = preco_estimado * 0.85, preco_estimado * 1.15

            # Exibição dos Resultados Técnicos
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor de Mercado Estimado", f"R$ {preco_estimado:,.2f}")
            c2.metric("Intervalo Admissível (Mín/Máx)", f"R$ {limite_inferior:,.2f} a R$ {limite_superior:,.2f}")
            c3.metric("Precisão do Modelo (R²)", f"{f'{r2_score*100:.2f}%' if r2_score > 0 else 'N/A'}")

            st.info(f"📐 **Variáveis processadas no cálculo multifatorial:** {', '.join(variaveis_independentes)}")

            # Gráfico de Dispersão
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
            st.warning(f"⚠️ Amostras insuficientes após tratamento. Linhas válidas: {len(df)}.")
else:
    st.info("💡 Escolha uma região salva no menu lateral ou selecione a opção de upload manual para começar.")
