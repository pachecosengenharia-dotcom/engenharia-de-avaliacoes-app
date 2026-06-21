from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# 1. Limpeza mais robusta
def clean_numeric(x):
    return pd.to_numeric(x.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')

# 2. Verificar VIF (Para eliminar variáveis redundantes)
def check_vif(X):
    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    return vif_data

# No fluxo:
df_modelo = df[features + [col_alvo]].apply(clean_numeric)
# ... após o dropna ...
# Recomendo usar statsmodels para o modelo final, pois ele fornece o R² ajustado e p-values 
# que são exigidos em laudos de Engenharia de Avaliações.
