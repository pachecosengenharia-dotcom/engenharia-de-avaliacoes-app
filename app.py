import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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
st.markdown("Modelagem estatística adaptativa e laudos técnicos em conformidade com as diretrizes normativas.")

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

# --- PROCESSAMENTO TÉCNICO ADAPTATIVO ---
if df is not None:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    def remover_acentos(texto):
        return ''.join(c for c in unicodedata.normalize('NFD', str(texto)) if unicodedata.category(c) != 'Mn')
    
    df.columns = [remover_acentos(col).strip() for col in df.columns]

    colunas_possiveis = {
        'Preco': ['Preco', 'Valor', 'Vlr', 'PRECO', 'Valor Total', 'Valor_Total', 'Valor Unitario'],
        'Area_Construida': ['Area_Construida', 'Area Construida', 'Area Const', 'Area Privativa', 'Area Util', 'M2 Privativa'],
        'Area_Terreno': ['Area_Terreno', 'Area Terreno', 'Terreno', 'AREA_TERRENO', 'Area do Terreno'],
        'Quartos': ['Quartos', 'Dormitorios', 'QUARTOS', 'Dormitorios Total'],
        'Suites': ['Suites', 'SUITES', 'Suite', 'SUITE'],
        'Vagas': ['Vagas', 'Garagem', 'VAGAS', 'Vagas Garagem'],
        'Conservacao': ['Conservacao', 'CONSERVACAO', 'Estado de Conservacao', 'Conservacao Imovel'],
        'Padrao_Acabamento': ['Padrao_Acabamento', 'Padrao', 'Acabamento', 'PADRAO', 'Padrao de Acabamento'],
        'Setor_Urbano': ['Setor_Urbano', 'Setor Urbano', 'Setor', 'SETOR', 'SETOR_URBANO', 'Setor urbano'],
        'Data_Evento': ['Data_Evento', 'Data Evento', 'Data', 'DATA', 'Data do Evento'],
        'Evento': ['Evento', 'EVENTO', 'Fator Evento'],
        'Idade_Aparent': ['Idade Aparente', 'Idade_Aparent', 'Idade']
    }

    for padrao, sinonimos in colunas_possiveis.items():
        for col in df.columns:
            if col in sinonimos:
                df = df.rename(columns={col: padrao})

    if 'Preco' not in df.columns or 'Area_Construida' not in df.columns:
        st.error(f"Não conseguimos mapear as colunas essenciais nesta planilha. Cabeçalhos atuais: {list(df.columns)}")
    else:
        todas_vars_possiveis = ['Area_Construida', 'Area_Terreno', 'Quartos', 'Suites', 'Vagas', 'Conservacao', 'Padrao_Acabamento', 'Setor_Urbano', 'Data_Evento', 'Evento', 'Idade_Aparent']
        variaveis_independentes = [v for v in todas_vars_possiveis if v in df.columns]

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
                label_nome = "Quartos" if var == 'Quartos' else ("Suítes" if var == 'Suites' else "Vagas de Garagem")
                caracteristicas_avaliando[var] = st.sidebar.slider(f"Quantidade de {label_nome}", 0, 5, 1 if var != 'Quartos' else 3)
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
                caracteristicas_avaliando[var] = st.sidebar.number_input("Fator de Evento", value=1.0, step=0.05)
            elif var == 'Idade_Aparent':
                caracteristicas_avaliando[var] = st.sidebar.number_input("Idade Aparente (Anos)", value=10.0, step=1.0)

        if len(df) >= len(variaveis_independentes) + 2:
            X = df[variaveis_independentes]
            y = df['Preco']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            
            modelo = LinearRegression().fit(X_scaled, y.values)
            
            y_pred_todo = modelo.predict(X_scaled)
            dados_imovel_lista = [caracteristicas_avaliando[var] for var in X.columns]
            dados_imovel_scaled = scaler.transform(np.array([dados_imovel_lista]))
            
            preco_estimado = max(0, modelo.predict(dados_imovel_scaled)[0])
            r2_score = modelo.score(X_scaled, y.values)
            
            limite_inferior, limite_superior = preco_estimado * 0.85, preco_estimado * 1.15

            # Diagnósticos de Cook e Resíduos
            residuos = y.values - y_pred_todo
            mse = np.mean(residuos ** 2) if np.mean(residuos ** 2) > 0 else 1.0
            leverage = np.ones(len(df)) * (len(variaveis_independentes) / len(df))
            distancia_cook = (residuos ** 2 / (len(variaveis_independentes) * mse)) * (leverage / (1 - leverage) ** 2)
            corte_cook = 4 / len(df)

            # Apresentação na Tela
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor de Mercado Estimado", f"R$ {preco_estimado:,.2f}")
            c2.metric("Intervalo Admissível (Mín/Máx)", f"R$ {limite_inferior:,.2f} a R$ {limite_superior:,.2f}")
            c3.metric("Precisão do Modelo (R²)", f"{r2_score*100:.2f}%")

            # RECONSTRUÇÃO LIMPA DA EQUAÇÃO MATEMÁTICA (Sem dízimas científicas ou poluição visual)
            coef_originais = modelo.coef_ / scaler.scale_
            intercept_original = modelo.intercept_ - np.sum(modelo.coef_ * scaler.mean_ / scaler.scale_)
            
            equacao_texto = f"Preço = {intercept_original:,.2f}"
            for var, coef in zip(X.columns, coef_originais):
                sinal = "+" if coef >= 0 else "-"
                equacao_texto += f" {sinal} {abs(coef):,.2f} × {var}"
            
            st.info(f"📐 **Equação de Regressão Linear Múltipla:** \n`{equacao_texto}`")

            # --- MATRIZ DE DIAGNÓSTICOS GRÁFICOS ---
            fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

            # 1. Dispersão Real
            axs[0,0].scatter(df['Area_Construida'].values, df['Preco'].values, color='#002d62', alpha=0.5, label="Amostras")
            axs[0,0].scatter([caracteristicas_avaliando['Area_Construida']], [preco_estimado], color='#d9534f', s=130, marker='*', zorder=5, label="Avaliando")
            axs[0,0].set_title("Dispersão: Preço vs Área", fontsize=10, weight='bold', color='#002d62')
            axs[0,0].set_xlabel("Área Construída (m²)")
            axs[0,0].set_ylabel("Preço (R$)")
            axs[0,0].legend(fontsize=8)

            # 2. Aderência Real
            axs[0,1].scatter(y.values, y_pred_todo, color='#002d62', alpha=0.5)
            axs[0,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5, label="Aderência Ideal")
            axs[0,1].set_title("Gráfico de Aderência (Real vs Estimado)", fontsize=10, weight='bold', color='#002d62')
            axs[0,1].set_xlabel("Preço Real (R$)")
            axs[0,1].set_ylabel("Preço Estimado (R$)")
            axs[0,1].legend(fontsize=8)

            # 3. Cook
            axs[1,0].stem(np.arange(len(distancia_cook)), distancia_cook, markerfmt=' ', linefmt='#002d62')
            axs[1,0].axhline(corte_cook, color='#d9534f', linestyle='--', lw=1.5, label="Limite (4/n)")
            axs[1,0].set_title("Distância de
