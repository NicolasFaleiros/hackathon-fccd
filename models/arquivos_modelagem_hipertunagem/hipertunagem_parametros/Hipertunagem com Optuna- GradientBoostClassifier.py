import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
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


threshold = 0.5
k = 10  # Número de folds ajustado para otimização


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

    modelo = GradientBoostingClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        random_state=0
    )

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

    return media_brier_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Alterado para 100 a 1000
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),  # Alterado para 0.001 a 0.1
        'max_depth': trial.suggest_int('max_depth', 3, 15),  # Alterado para 3 a 15
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),  # Alterado para 1 a 10
        'max_features': trial.suggest_float('max_features', 0.1, 1.0)  # Alterado para 0.1 a 1.0
    }
    
    folds = preparar_dados(X=X, y=y, k=k)
    media_brier_score = validacao_cruzada(folds=folds, threshold=threshold, params=params, df_teste=df_teste)
    
    # Report the score for pruning
    trial.report(media_brier_score, step=0)
    
    # Prune trial if needed
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return media_brier_score

print('Iniciando importação')
tempo_inicio = time.time()

df_treino = pd.read_csv('arquivos_treino_teste\Formato_por_dia\df_treino_completo_3dias')
df_teste = pd.read_csv('arquivos_treino_teste\Formato_por_dia\df_teste_completo_3dias.csv')

# Remover a coluna id_participante dos dados de treino
colunas_para_dropar = ['resultado', 'id_participante']
X = df_treino.drop(columns=colunas_para_dropar, axis=1).values
y = df_treino['resultado'].values

# df_treino = pd.read_csv('arquivos_treino_teste/df_treino_completo.csv')
# df_teste = pd.read_csv('arquivos_treino_teste/df_teste_completo.csv')

# # Remover a coluna id_participante dos dados de treino
# colunas_para_dropar = [ 'id_lance', 'leilao', 'mercadoria', 'tempo', 'resultado', 'id_participante', 'media_urls_porip',
#                        'qtde_mercadorias_participante', 'mediana_paises_por_ip', 'ratio_media_mediana_lances_porleilao', 'std_lances_pico',
#                        'std_lances_fora_pico', 'tempo_primeiro_lance', 'tempo_ultimo_lance', 'tem_lance_todo_dia']

# X = df_treino.drop(columns=colunas_para_dropar, axis=1).values
# y = df_treino['resultado'].values

# colunas_para_dropar_teste = [ 'id_lance', 'leilao', 'mercadoria', 'tempo','media_urls_porip', 'qtde_mercadorias_participante',
#                              'mediana_paises_por_ip', 'ratio_media_mediana_lances_porleilao', 'std_lances_pico', 'std_lances_fora_pico',
#                              'tempo_primeiro_lance', 'tempo_ultimo_lance', 'tem_lance_todo_dia']

# df_teste.drop(columns=colunas_para_dropar_teste, axis=1, inplace=True)


tempo_fim = time.time()
print(f'Esse processo demorou {tempo_fim - tempo_inicio:.2f} segundos')

# Instâncias do SimpleImputer e StandardScaler
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# Optuna study para otimização dos hiperparâmetros com pruner e sampler
study = optuna.create_study(direction='minimize', 
                            sampler=TPESampler(), 
                            pruner=MedianPruner())
study.optimize(objective, n_trials=30)

print("Número de trials: ", len(study.trials))
print("Melhores hiperparâmetros: ", study.best_params)
print("Melhor Brier Score: ", study.best_value)

melhores_parametros = study.best_params
print("Melhores hiperparâmetros: ", melhores_parametros)

with open("melhores_parametros_gb2.json", "w") as arquivo:
    json.dump(melhores_parametros, arquivo)
