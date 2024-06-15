# Projeto Detecção de Fraudes - Hackathon - FCCD

Projeto de detecção de fraude desenvolvido durante o curso de Formação Completa de Cientista de Dados baseado na competição Kaggle [Facebook Recruiting IV](https://www.kaggle.com/competitions/facebook-recruiting-iv-human-or-bot/overview). Link para competição FCCD [aqui](https://www.kaggle.com/competitions/hackathon-01-fccd). Dados originais disponíveis no Kaggle para download. 
O modelo está otimizando a métrica Brier Score por ser a métrica alvo definida na competição. Nota final: 

## 💭 Ideia utilizada

A ideia para desenvolvimento desse modelo foi criar padrões/features para identificar usuários fraudulentos em vez de tentar identificar lances fraudulentos de forma separada. Além disso, encontrar um padrão de dias na distribuição de lances permitiu a divisão do comportamento dos usuários dia-a-dia aumentando o tamanho de database de treino e melhorando o desempenho do modelo.

## ⚙️ Features

As informações originais foram:

**Coluna** | **Explicação**
----------- | ------------
id_lance | Identificador único do lance
id_participante | Identificador único do participante
leilao | Identificador único do leilão
mercadoria | A categoria da mercadoria leiloada
dispositivo | O dispositivo utilizado pelo visitante
tempo | O tempo que o lance foi feito
pais | O país que o IP pertence
ip | O IP do participante
url | A URL de onde o participante foi referido


**Foram criadas 199 features no projeto, exemplos:**

* Features baseadas em métricas das informações iniciais, como total, média, mediana de lances e de lances de um usuário.
* Combinações como Dispositivos por IP, URLs por leilão, entre outras.
* Cálculos de entropia de informações em cada dia
* Features baseadas no tempo, como o quartil de tempo que o lance foi enviado, média/mediana de diferença de tempo entre lances e se um lance foi feito ou não durante o pico do leilão 
* Ratios e produtos de features criadas. Foi feito um feature importance durante a criação do modelo e combinando (por meio de divisão ou produto) as features que mais influenciavam os resultados.

### **Importância das features (maior para menor):**: 

As 10 primeiras:
feature | valor_importancia
--------|---------
mediana_lances_pordispositivo | 0.0951
media_urls_porleilao | 0.0395
ratio_media_mediana_dispositivos_porleilao | 0.0320
media_lances_pordispositivo | 0.0308
dispositivos_duplicados | 0.0267
porcentagem_lances_quartil_2 | 0.0221
media_dispositivos_porleilao | 0.0184
ratio_max_media_urls_porip | 0.0181
ips_duplicados | 0.0180
paises_duplicados | 0.0176


## Preparação de dados e Validação Cruzada

### Validação Cruzada

A validação cruzada utilizada foi StratifiedKFold da biblioteca sklearn.model_selection, com um total de 100 folds 

### Preparação de dados

Utilizando SimpleImputer da biblioteca sklearn.impute e StandardScaler de sklearn.preprocessing os dados foram preparados à cada fold antes do treinamento

```
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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

```

## 📈 Informações sobre algoritmo/hipertunagem/balanceamento

### Algoritmo utilizado:
Foi utilizado o classificador GradientBoostingClassifier da biblioteca sklearn.ensemble.
```
from sklearn.ensemble import GradientBoostingClassifier

modelo_final = GradientBoostingClassifier(
    max_depth=melhores_parametros['max_depth'],
    n_estimators=melhores_parametros['n_estimators'],
    learning_rate=melhores_parametros['learning_rate'],
    max_features=melhores_parametros['max_features'],
    min_samples_split=melhores_parametros['min_samples_split'],
    min_samples_leaf=melhores_parametros['min_samples_leaf'],
    random_state=0
)
```

### Hipertunagem
Os parâmetros foram otimizados pela biblioteca por optuna utilizando esse range:
```
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),  
        'max_depth': trial.suggest_int('max_depth', 3, 15),  
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10), 
        'max_features': trial.suggest_float('max_features', 0.1, 1.0)  
    }

```

**parâmetro** | **valor**
------------- | -----------
n_estimators | 143
learning_rate | 0.017876949339318857
max_depth | 12
min_samples_split | 19
min_samples_leaf | 10
max_features | 0.3444282296052409

### Balanceamento utilizado:
Não foi utilizada nenhuma técnica específica de balanceamento, dado que o algoritmo GradientBoostingClassifier já lida bem com dados desbalanceados.

## 🔧 Pré-requisitos

Acesse o arquivo requirements.txt do repositório

## 📋 Divisão do código

Por questões de capacidade computacional, foram divididas algumas partes do código, como Feature Engineering.

A sequência de execução para atingir os mesmos resultados é:
1. divisao-lances-por-dia
2. feature-engineering-por-dia
3. juncao-df-dias
4. Hipertunagem com Optuna- GradientBoostClassifier
5. modelagem

## 📦 Implantação

Após execução do arquivo modelagem.py, um arquivo para submission irá ser criado na pasta arquivos_submission no formato 'resultado_GradientBoostingClassifier.csv_hipertunado_com_optuna_{k}folds.csv'

Modelo de READ.me criado por: [Armstrong Lohãns](https://gist.github.com/lohhans)
