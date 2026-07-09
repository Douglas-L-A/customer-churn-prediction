import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'logistic'):
    """
    Treina o modelo de machine learning especificado.
    """
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Modelo {model_type} não suportado.")
        
    print(f"Treinando modelo: {model_type}...")
    model.fit(X_train, y_train)
    print("Treinamento concluído.")
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Avalia o modelo utilizando métricas padrão de classificação.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    print("Métricas de Avaliação:")
    for metric, value in metrics.items():
        print(f" - {metric.capitalize()}: {value:.4f}")
        
    return metrics

def save_model(model, filepath: str):
    """
    Salva o modelo treinado em um arquivo.
    """
    joblib.dump(model, filepath)
    print(f"Modelo salvo em {filepath}")

def load_model(filepath: str):
    """
    Carrega um modelo treinado.
    """
    model = joblib.load(filepath)
    print(f"Modelo carregado de {filepath}")
    return model
