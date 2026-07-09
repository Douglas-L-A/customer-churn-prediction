import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def main():
    # Caminhos
    data_path = 'data/processed/telco_churn_processed.csv'
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'logistic_regression.joblib')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(f"Lendo dados de {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separando X e y
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    print("Treinando o modelo de Regressão Logística em todos os dados processados...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    # Salvando as features esperadas para uso futuro na inferência
    expected_features = list(X.columns)
    
    print(f"Salvando modelo em {model_path}...")
    # Salvaremos um dicionário contendo o modelo e as features esperadas
    joblib.dump({'model': model, 'features': expected_features}, model_path)
    
    print("Treinamento finalizado com sucesso!")

if __name__ == "__main__":
    main()
