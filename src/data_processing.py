import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega o dataset a partir do caminho especificado.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dados carregados com sucesso de {filepath}")
        return df
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None

def preprocess_input_for_inference(df_input: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """
    Pré-processa os dados brutos vindos da interface web (Streamlit) para o formato do modelo.
    Cria variáveis dummy e alinha com as features que o modelo usou no treinamento.
    """
    # 1. Transformar as variáveis categóricas recebidas
    df_encoded = pd.get_dummies(df_input)
    
    # 2. Reindexar para garantir que todas as colunas do treino estejam presentes, 
    # preenchendo as que faltam com 0/False (útil pois a UI não pede todos os 30 campos)
    df_final = df_encoded.reindex(columns=expected_features, fill_value=False)
    
    # Colunas numéricas ausentes (ex: SeniorCitizen) ficam com valor 0 por causa do False
    return df_final

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza e o pré-processamento dos dados.
    - Preenchimento de nulos
    - Encoding de variáveis categóricas
    - Escalonamento de features
    """
    df_processed = df.copy()
    
    # TODO: Implementar lógica de pré-processamento específica do dataset
    # Exemplo: df_processed.dropna(inplace=True)
    
    print("Pré-processamento concluído.")
    return df_processed
