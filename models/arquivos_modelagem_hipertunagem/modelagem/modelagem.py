import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from warnings import filterwarnings
import time
import json


filterwarnings('ignore', category=UserWarning)

pd.set_option('display.max_rows', None)

def preparar_dados(X, y, k):
    global imputer, scaler

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    folds = []

    for fold_idx, (train_index, test_index) in enumerate(cv.split(X, y)):
        print(f"Preparando fold {fold_idx + 1}...")
        start_time = time.time()
        X_treino, X_teste = X[train_index], X[test_index]
        y_treino, y_teste = y[train_index], y[test_index]

        # Imputação de valores ausentes
        X_treino = imputer.fit_transform(X_treino)
        X_teste = imputer.transform(X_teste)

        # Aplicação do StandardScaler nos dados de treino e teste
        X_treino = scaler.fit_transform(X_treino)
        X_teste = scaler.transform(X_teste)

        end_time = time.time()
        print(f"Tempo para preparar dados no fold {fold_idx + 1}: {end_time - start_time:.2f} segundos")
        folds.append((X_treino, y_treino, X_teste, y_teste))

    print("Preparação dos dados concluída.")
    return folds

def validacao_cruzada(folds, threshold, params, df_teste):
    global imputer, scaler

    print(f"Iniciando validação cruzada para o modelo Gradient Boosting...")
    brier_scores = []

    modelo = modelo_final

    for fold_idx, (X_treino, y_treino, X_teste, y_teste) in enumerate(folds):
        print("=-" * 6 + f"Fold: {fold_idx + 1}" + "-=" * 6)
        start_time = time.time()

        modelo.fit(X_treino, y_treino)

        y_pred_proba = modelo.predict_proba(X_teste)[:, 1]
        brier_score = brier_score_loss(y_teste, y_pred_proba)
        brier_scores.append(brier_score)
        print(f"Brier Score: {brier_score:.8f}")

        end_time = time.time()
        print(f"Tempo para treinar modelo Gradient Boosting no fold {fold_idx + 1}: {end_time - start_time:.2f} segundos")

    media_brier_score = np.mean(brier_scores)
    std_brier_score = np.std(brier_scores)
    print(f"Média do Brier Score no modelo Gradient Boosting: {media_brier_score:.8f} +/- {std_brier_score:.8f}")

    # Aplicando no df_teste
    X_submission = df_teste.drop(columns=['id_participante'])

    # Imputação e escalonamento dos dados de teste
    X_submission = imputer.transform(X_submission)
    X_submission = scaler.transform(X_submission)

    y_pred_proba = modelo.predict_proba(X_submission)[:, 1]
    y_pred = np.where(y_pred_proba > threshold, 1, 0)

    df_resultado = pd.DataFrame({'id_participante': df_teste['id_participante'], 'resultado': y_pred})

    nomedo_modelo = modelo.__class__.__name__
    nome_arquivo = f"resultado_{nomedo_modelo}.csv"

    arquivo_submission = df_resultado.groupby('id_participante')['resultado'].apply(
        lambda x: 1 if any(x == 1) else 0).reset_index()
    
    arquivo_submission = arquivo_submission.drop_duplicates(subset='id_participante', keep='first')
    
    arquivo_submission.to_csv(f'arquivos_submission\\{nome_arquivo}_hipertunado_com_optuna_{k}folds.csv', index=False)

    return media_brier_score


print('Iniciando importação')
tempo_inicio = time.time()

df_treino = pd.read_csv('arquivos_treino_teste\Formato_por_dia\df_treino_completo_3dias.csv')
df_teste = pd.read_csv('arquivos_treino_teste\Formato_por_dia\df_teste_completo_3dias.csv')

# Remover a coluna id_participante dos dados de treino
colunas_para_dropar = ['resultado', 'id_participante']
X = df_treino.drop(columns=colunas_para_dropar, axis=1).values
y = df_treino['resultado'].values


tempo_fim = time.time()
print(f'Esse processo demorou {tempo_fim - tempo_inicio:.2f} segundos')

# Instâncias do SimpleImputer e StandardScaler
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

threshold = 0.5
k = 100  # Número de folds


with open('arquivos_treino_teste\melhores_parametros_gb.json', "r") as arquivo:
    melhores_parametros = json.load(arquivo)


# Instanciando o modelo final com os melhores parâmetros
modelo_final = GradientBoostingClassifier(
    max_depth=melhores_parametros['max_depth'],
    n_estimators=melhores_parametros['n_estimators'],
    learning_rate=melhores_parametros['learning_rate'],
    max_features=melhores_parametros['max_features'],
    min_samples_split=melhores_parametros['min_samples_split'],
    min_samples_leaf=melhores_parametros['min_samples_leaf'],
    random_state=0
)

folds = preparar_dados(X=X, y=y, k=k)

melhor_brier_score = validacao_cruzada(folds=folds, threshold=threshold, params=melhores_parametros, df_teste=df_teste)

print(f"Melhor Brier Score após otimização: {melhor_brier_score:.8f}")