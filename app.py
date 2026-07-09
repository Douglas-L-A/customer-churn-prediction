import streamlit as st
import pandas as pd
import random
import joblib
from src.data_processing import preprocess_input_for_inference

# Configuração da página
st.set_page_config(
    page_title="Predição de Churn",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Dashboard de Retenção de Clientes")
st.markdown("""
Esta aplicação utiliza aprendizado de máquina para prever a probabilidade de um cliente cancelar o serviço (Churn).
Preencha as informações do cliente na barra lateral para obter a análise.
""")

# Barra Lateral - Entrada de Dados
st.sidebar.header("📊 Dados do Cliente")

def user_input_features():
    tenure = st.sidebar.slider("Tempo como Cliente (Meses)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Mensalidade ($)", min_value=15.0, max_value=150.0, value=65.0)
    
    # Calculando os gastos totais automaticamente
    total_charges = tenure * monthly_charges
    st.sidebar.markdown(f"**Gastos Totais Calculados:** ${total_charges:.2f}")
    
    st.sidebar.markdown("---")
    
    contract = st.sidebar.selectbox("Tipo de Contrato", ("Month-to-month", "One year", "Two year"))
    internet_service = st.sidebar.selectbox("Serviço de Internet", ("DSL", "Fiber optic", "No"))
    payment_method = st.sidebar.selectbox("Método de Pagamento", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
    
    st.sidebar.markdown("---")
    
    gender = st.sidebar.selectbox("Gênero", ("Male", "Female"))
    partner = st.sidebar.selectbox("Possui Parceiro?", ("Yes", "No"))
    paperless = st.sidebar.selectbox("Fatura Digital?", ("Yes", "No"))
    online_security = st.sidebar.selectbox("Segurança Online?", ("Yes", "No", "No internet service"))

    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method,
        'gender': gender,
        'Partner': partner,
        'PaperlessBilling': paperless,
        'OnlineSecurity': online_security
    }
    return pd.DataFrame(data, index=[0])

df_customer = user_input_features()

# Painel Principal
st.subheader("Resumo do Cliente")
st.write(df_customer)

st.markdown("---")

if st.button("🔮 Analisar Risco de Cancelamento (Churn)", type="primary"):
    with st.spinner('Processando os dados no modelo...'):
        
        # 1. Carrega o dicionário contendo o modelo e as features esperadas
        model_dict = joblib.load('models/logistic_regression.joblib')
        model = model_dict['model']
        expected_features = model_dict['features']
        
        # 2. Transforma o DataFrame do formulário nas 30 colunas que o modelo espera
        df_processed = preprocess_input_for_inference(df_customer, expected_features)
        
        # 3. Calcula a probabilidade matemática do cliente pertencer à classe 1 (Churn)
        probabilidade_churn = model.predict_proba(df_processed)[0][1]
        
        # Exibição do Resultado
        if probabilidade_churn > 0.5:
            st.error(f"⚠️ **Risco ALTO de Cancelamento!**")
            st.write(f"A probabilidade de este cliente cancelar o serviço é de **{probabilidade_churn:.1%}**.")
            st.warning("Recomendação: Entre em contato imediato e ofereça incentivos para a transição a contratos anuais.")
        else:
            st.success(f"✅ **Risco BAIXO de Cancelamento**")
            st.write(f"A probabilidade de este cliente cancelar o serviço é de apenas **{probabilidade_churn:.1%}**.")
            st.info("Recomendação: Continue monitorando os indicadores de satisfação.")
