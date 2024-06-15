
# Hackathon FCCD 2024

## Objetivos e entregáveis:
- Executar uma Análise Exploratória de Dados;
- Confeccionar um modelo de Machine Learning para prever usuários robôs;
- Demonstrar o impacto desses usuários no contexto do negócio;
- 5 perguntas de negócios mais relevantes; e
- Um Dashboard.

O desafio imposto aos Cientistas de Dados foi de detectar se um lance é uma fraude (se foi feito por robô ou não), elaborando um modelo de Machine Learning. Após diversos testes com esta perspectiva, percebemos que um lance só pode ser fraudulento se o usuário que o originou também for fraudulento. Esta percepção permitiu um Feature Engineering melhor para aplicação e teste dos modelos.

O desafio imposto aos Analistas de Dados foi de realizar uma análise exploratória de dados e responder a dez perguntas de negócios relevantes, somente 5 destas perguntas seriam apresentadas à banca avaliadora, entregando um dashboard que apresente as principais descobertas.

## Apêndice

- Entendimento do contexto de negócio
- Análise Exploratória de Dados
- Questões de negócio relevantes
- Modelo de Machine Learning
- Tunagem de Hiperparâmetros
- Recomendações e Ganhos


## Entendimento do Contexto de Negócio

Leilão é uma modalidade de negociação envolvendo diversos tipos de produtos, desde bens de consumo até artigos de luxo, sendo mediado pela figura do leiloeiro. O processo inicia com um lance mínimo inicial, e de forma crescente, vai subindo os valores à medida que cresce o número de lances.
- Leilões podem variar em nível de competitividade, a depender de variáveis como item leiloado, duração dos lances, segmento etc.
- Participação simultânea de usuários realizando lances para obtenção dos produtos
- Possibilidade de usuários fraudulentos fazendo uso de automações e técnicas como bidding spinning
- Maioria dos leilões online apresentam termos claros proibindo o uso de robôs pelos usuários, garantindo um campo de jogo mais justo

## Análise Exploratória de Dados

A base “lances.csv” possui mais de 7.5mi de lances, não existem lances repetidos e os únicos nulos estão na coluna país (quase 9mil lances sem informação). Algumas colunas possuem informações criptografadas (por caracteres alfanuméricos) para proteger as informações dos usuários. A coluna tempo possui um padrão diferente envolvendo somente números e assumimos que a ordem e escala relativa do tempo se manteve. A coluna país possui os nomes de cada país escritos tanto abreviados como por extenso, tratamos isto padronizando todos por extenso.

O conjunto de treino “train.csv” possui o rótulo de resultado que define usuários fraudulentos e comuns. Apenas cerca de 5% dos usuários são robôs, é esperado este tipo de situação visto que é natural supor que somente uma minoria dos usuários serão infratores. Este tipo de desbalanceamento deve ser tratado para a modelagem de algoritmos preditivos de machine learning  O algoritmo utilizado (GradientBoostingClassifier possui um desbalanceamento dentro do algoritmo).

Explorando, percebemos que existem 6614 usuários que realizaram lances, porém, alguns estão presentes nas bases de treino e teste, mas não estão na base de lances. Utilizando o rótulo “resultado” (da base “train.csv”) e mesclando com a base de lances resulta em um filtro de mais de 2.5mi de lances que correspondem à somente 12.199 leilões. Neste filtro, de 1393 usuários (94.97%), há apenas 70 (5.03%) fraudadores que correspondem à 13.15% dos lances fraudulentos na base “lances.csv”.

Ainda na base lances mesclada com o rótulo “resultado”, as variáveis categóricas não criptografadas correspondem aos países (196 países), dispositivos (5393 dispositivos diferentes utilizados por usuários) e mercadoria (9 tipos de produtos leiloados).

Considerando volume de lances, a mercadoria mais comum é a de artigos esportivos, o dispositivo mais utilizado é o “phone4” e o país com mais lances é a Índia. A Índia possui a maior proporção de usuários fraudadores (cerca de 18%). 7 de 196 países possuem 100% de fraude. Além disso, Macau também possui 99% de lances fraudulentos, realizados por um único robô. Outros países também possuem alto volume de lances e alta proporção de fraudes, em especial Japão, Coreia do Sul, Alemanha, Áustria, Canadá, Austrália e Estados Unidos. Porém em todos os casos somente uma pequena parcela de usuários foram responsáveis pelas fraudes. Cerca de 9.5% (513) dos dispositivos só possuem lances oriundos de usuários que utilizam robôs, embora o volume de lances envolvendo esses dispositivos tende a ser baixo. 69% dos lances envolvendo o produto computador foram feitos por robôs. Além disso, um valor gigantesco de lances envolvendo artigos esportivos foram feitos por robôs (198.246 lances).

#### Países escolhidos após plotagem em gráficos:
- Macau: 1 usuário fraudulento realizou 140 lances.
- Japão: 12 usuários fraudulentos realizaram 4.857 lances.
- Israel: 10 usuários fraudulentos realizaram 1.317 lances.
- Canadá: 32 usuários fraudulentos realizaram 6.822 lances.
- Coreia do Sul: 4 usuários fraudulentos realizaram 6.340 lances.
- Suíça: 18 usuários fraudulentos realizaram 1.738 lances.
- Suécia: 17 usuários fraudulentos realizaram 1.890 lances.
- Alemanha: 34 usuários fraudulentos realizaram 14.595 lances.
- Áustria: 12 usuários fraudulentos realizaram 2.960 lances.
- Austrália: 38 usuários fraudulentos realizaram 7.094 lances.
- Estados Unidos: 45 usuários fraudulentos realizaram 54.434 lances.

Investigando os dispositivos com 100% de lances oriundos de robôs na base lances mesclada com o rótulo “resultado”, o que mais fez lances foi o “phone3243” com 170 lances realizados por somente 3 usuários. Apesar dos países com 100% de lances feitos por robôs possuírem poucos lances realizados existem exceções como Japão e Coreia do Sul com mais de 90% de lances feitos por usuários fraudulentos em milhares de lances realizados.

## Questões de Negócio Relevantes


- **É comum a participação de um mesmo usuário em leilões diferentes? A participação de um mesmo usuário em diversos leilões, pode ser um indicativo de ser um não humano?**
    - Apesar de apresentar um percentual elevado de fraudadores com participação em mais de um leilão, o público em geral, participantes de leilões, costumam participar de mais de 1 leilão.

- **Quais os segmentos de mercadorias leiloadas com a maior taxa de incidência de robôs?**
    - Evidencia-se que ao passo que determinados segmentos como vestuário e peças de automóveis apresentam uma taxa de incidência de robôs nulas, segmentos como computadores e artigos esportivos apresentam uma taxa de 69% e 18%, respectivamente.

- **Usuários tendem a apresentar mais de um tipo de dispositivo para participação em um mesmo leilão?**
    - Em média, considerando toda a amostra de dados, temos que 63.68% dos usuários utilizaram mais de um dispositivo. Quando considerado o público da amostra categorizada como robô, esse percentual sobe para 88.57%.

- **Usuários costumam apresentar uma única conta de pagamento para participação em leilões?**
    - Apesar de normalmente acessarem os leilões por meio de mais de um dispositivo, não apresentam contas de pagamento diferentes.

- **Existe um volume de lances específico que caracteriza um comportamento fraudulento?**
    - Por meio da observação dos quartis da contagem média de lances por participante, verifica-se que enquanto os 75% dos usuários geral apresentam uma média de até 210 lances, o público limitado aos participantes classificados como robôs apresentam média superior aos 2900 lances.

- **Usuários humanos apresentam um tempo de lance diferente dos usuários categorizados como robôs?**
    - Sim. A variável tempo está “criptografada”, mas assumindo que as propriedades de ordem e escala relativa foram preservadas, então há uma diferença significativa entre lances de usuários humanos e de robôs.
    - Um usuário humano demora, em média, 3 vezes mais entre um lance e outro do que um usuário classificado como robô. Os usuários legítimos também possuem uma mediana 0.5 vezes maior, um desvio-padrão 18 vezes maior e um valor máximo entre um lance e outro 213 vezes maior, o que sugere que, de fato, os usuários humanos tendem a demonstrar um período maior entre um lance e outro.

- **Existem URLs em leilões mais suscetíveis a fraudes?**
    - Sim. Cerca de 8% dos URLs possuem 100% (43.107) de incidência de fraude. Desse último valor, 84% correspondem a URLs que foram utilizadas somente uma vez.
    - A média de vezes que essas URLs foram utilizadas é de 1.8, mas o desvio-padrão é de 10. Isso ocorre pois algumas URLs tiveram centenas de lances fraudulentos.
    - O caso mais fora da curva foi de uma URL com 937 lances, realizados por somente 2 usuários fraudulentos em leilões diferentes.

- **Um usuário fraudulento tende a fazer mais lances do que um usuário legítimo?**
    - Sim. Em média, a quantidade máxima de lances de um usuário fraudulento é 5.7 vezes maior que a de um usuário humano, e a mediana 40 vezes maior.
    - As fraudes são realizadas por poucos usuários, mas em geral eles possuem um comportamento bem agressivo em termos de quantidade de lances por leilão. O robô quer ganhar o leilão, então é natural pensar que irá realizar a quantidade de lances que for necessário para ganhar.

- **Existe alguma relação entre o dispositivo usado e a incidência de fraudes?**
    - Alguns dispositivos possuem 100% de incidência de fraude (513), porém, em geral, foram usados poucas vezes e por poucos usuários.
    - A média é de 23% de incidência de fraude por modelo de dispositivo. Por outro lado, a mediana é de 1.7%, o que indica que a média é fortemente influenciada por outliers. Podemos confirmar isso olhando para o desvio-padrão de 36% e para os quase 10% de dispositivos com 100% de lances oriundo de usuários fraudulentos.
    - Para testar se há uma associação ou dependência entre o tipo de dispositivo usado e a ocorrência de fraudes usaremos o teste qui-quadrado. O teste qui-quadrado de independência testa a hipótese nula de que as duas variáveis são independentes.
    - Para um teste com grau de significância de 5%, o valor obtido foi de “993981” e o P-valor < 0.05, com 5392 graus de liberdade. Portanto, rejeitamos a hipótese nula, isto é, acreditamos que existe uma relação significativa entre o modelo do dispositivo e a incidência de fraudes.
    - Como o tamanho da amostra é relativamente grande, o teste pode ter captado diferenças muito pequenas que são estatisticamente significativas, mas não necessariamente significativas do ponto de vista prático. Para avaliar a magnitude da associação, calculamos o tamanho do efeito (V de Cramer) e obtivemos 0.62 como resultado, o que sugere de fato uma forte associação.

- **Qual é a taxa geral de fraudes nos leilões?**
    - Cerca de 13% dos lances foram realizados por usuários detectados como robô. Por outro lado, cerca de 5% dos usuários conhecidos foram diagnosticados como robô.

## Modelo de Machine Learning

Após executar a EDA e reuniões em conjunto com os Analistas de Dados, pudemos afirmar que existiam informações faltantes de países, dados ausentes de usuários em treinos e teste. Tratava-se de usuários que não existiam nas bases lances, treino ou teste (e vice-versa). Os valores ausentes que pudessem ser preenchidos com a média, foram preenchidos.

Após diversos testes com variações nos modelos:
- **LGBM Classifier**
- **Árvore de Decisão**
    - Desbalanceando classes com SMOTE
- **Regressão Logística**
- **XGBoost**
- **XGB Classifier**
    - Desbalanceando classes com SMOTE, ADASYN e RUS
- **CatBoost Classifier**
    - Desbalanceando classes com SMOTE, ADASYN e RUS
- **Bagging Classifier**
    - Desbalanceando classes com SMOTE, ADASYN e RUS
    - Hiperparâmetros tunados com Optuna
- **Gradient Boosting Classifier**
    - Hiperparâmetros tunados com Optuna

Optamos pelo “GradientBoostingClassifier” por se tratar de um problema de classificação, o desempenho dele neste tipo de situação tende a ser superior criando um modelo de conjunto de árvores de decisão que são treinadas sequencialmente corrigindo os erros do modelo anterior. Se bem calibrado, resulta em modelos mais robustos e com melhor capacidade de generalização.

```bash
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

A ideia para desenvolvimento desse modelo foi criar padrões/features para identificar usuários fraudulentos em vez de tentar identificar lances fraudulentos de forma separada. Além disso, encontrar um padrão de dias na distribuição de lances permitiu a divisão do comportamento dos usuários dia-a-dia aumentando o tamanho de database de treino e melhorando o desempenho do modelo.

A métrica de avaliação definida pelo Hackathon da FCCD foi a Brier Score, que é uma medida de calibração das probabilidades previstas em problemas específicos de classificação, assim como funciona o RME para problemas de regressão, entre outros, há o Brier Score para problemas de classificação.

Após um extenso trabalho de Feature Engineering, decidimos que as seguintes features permaneceriam para treinar o modelo, foram criadas 199 features no projeto, exemplos:

- Features baseadas em métricas das informações iniciais, como total, média, mediana de lances e de lances de um usuário;
- Combinações como Dispositivos por IP, URLs por leilão, entre outras;
- Cálculos de entropia de informações em cada dia;
- Features baseadas no tempo, como o quartil de tempo que o lance foi enviado, média/mediana de diferença de tempo entre lances e se um lance foi feito ou não durante o pico do leilão; e
- Ratios e produtos de features criadas. Foi feito um feature importance durante a criação do modelo e combinando (por meio de divisão ou produto) as features que mais influenciavam os resultados.

**Importância das Features (maior para menor – 10 primeiras)**
| feature   | valor_importancia      |
| :---------- | :--------- |
| `mediana_lances_pordispositivo`      | 0.0951 |
| `media_urls_porleilao`      | 0.0395 |
| `ratio_media_mediana_dispositivos_porleilao`      | 0.0320 |
| `media_lances_pordispositivo`      | 0.0308 |
| `dispositivos_duplicados`      | 0.0267 |
| `porcentagem_lances_quartil_2`      | 0.0221 |
| `media_dispositivos_porleilao`      | 0.0184 |
| `ratio_max_media_urls_porip`      | 0.0181 |
| `ips_duplicados`      | 0.0180 |
| `paises_duplicados`      | 0.0176 |

#### Preparação de dados e Validação Cruzada
#### Validação Cruzada

A validação cruzada utilizada foi StratifiedKFold da biblioteca sklearn.model_selection, com um total de 100 folds

Preparação de dados
Utilizando SimpleImputer da biblioteca sklearn.impute e StandardScaler de sklearn.preprocessing os dados foram preparados à cada fold antes do treinamento.


```bash
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

## Tunagem de Hiperparâmetros

Os parâmetros foram otimizados pela biblioteca por optuna utilizando esse range:

```bash
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

Utilizamos a biblioteca Optuna para encontrar os melhores hiperparâmetros automaticamente a partir do modelo Gradient Boosting. Métrica de avaliação utilizada: Brier Score, em que se a probabilidade do usuário ser robô ultrapassar 0.5, é classificado como tal, abaixo disso é classificado como humano.

| Parâmetro   | Valor      |
| :---------- | :--------- |
| `n_estimators`      | 143 |
| `learning_rate`      | 0.017876949339318857 |
| `max_depth`      | 12 |
| `min_samples_split`      | 19 |
| `min_samples_leaf`      | 10 |
| `max_features`      | 0.3444282296052409 |

Com base na métrica “media_brier_score” e limites superiores/inferiores de hiperparâmetros definidos, o Optuna faz essa busca automática pelos melhores hiperparâmentros, comparando diferentes configurações entre eles. O objetivo é de encontrar os hiperparâmetros que resultaram na menor média de Brier Score, indicando o melhor desempenho do modelo em termos de calibração das probabilidades previstas.

#### Balanceamento utilizado:
Não foi utilizada nenhuma técnica específica de balanceamento, dado que o algoritmo GradientBoostingClassifier já lida bem com dados desbalanceados.

## Recomendações e Ganhos
O desenvolvimento acelerado de algoritmos de inteligência tem possibilitado uma revolução nos modelos preditivos de fraude, garantindo que ações possam ser implementadas visando atenuar os riscos e identificando usuários fraudadores. Aqui descrevemos ações com uso de IA que podem ser implementadas visando ganhos e redução de riscos.
Recomendações:
- Padronização da informação dos usuários;
- Implementar e monitorar o modelo em ambiente de produção, com atenção na escalabilidade e ajustando hiperparâmetros se necessário;
- Ajustar o modelo para tomar ações de interesse da empresa (notificar um time da empresa sobre os possíveis usuários fraudulentos ou simplesmente banir o usuário da plataforma);
 
