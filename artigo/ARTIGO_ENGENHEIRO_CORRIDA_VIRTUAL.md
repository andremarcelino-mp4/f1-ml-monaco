# Engenheiro de Corrida Virtual: aprendizado de máquina sobre telemetria do F1 23 para avaliar o efeito do acerto do carro no tempo de volta em Mônaco

**André Marcelino**
FECAF — Faculdade de Educação e Cultura Francisco Antônio Fernandes · Gestão de Tecnologia · Disciplina de Machine Learning / Deep Learning · 2026

Repositório: <https://github.com/andremarcelino-mp4/f1-ml-monaco> · Página do projeto: <https://f1-monaco-landing.vercel.app/>

---

## Resumo

Este trabalho investiga, em ambiente controlado de simulação, quanto da variação do tempo de volta no Circuito de Mônaco pode ser explicada pelo acerto do carro e quanto decorre de fatores de contexto e do próprio piloto. A telemetria UDP do jogo *F1 23* foi capturada em Python, gerando 277 voltas válidas (273 de Ferrari e Alfa Romeo, usadas na modelagem) e cerca de 1,54 milhão de amostras instantâneas. Foram testados onze acertos: uma base (S0), nove variações de um único fator (S1–S9) e um acerto multifatorial sugerido por inteligência artificial (S10). Sete algoritmos de regressão (Lasso, Ridge, ElasticNet, Random Forest, Extra Trees, Gradient Boosting e uma rede MLP exploratória, contra um *baseline* pela média) foram avaliados com validação cruzada agrupada por sessão (GroupKFold). A idade do pneu (≈ 43 % da importância consensual), o diferencial sem aceleração (≈ 26 %) e o carro (≈ 15 %) dominaram a explicação do tempo. O S10 reduziu o tempo médio em cerca de 1,5 s nos dois carros, enquanto o S2 (diferencial sem aceleração em 80) foi o pior (+1,93 s na Alfa Romeo). Após ajuste de hiperparâmetros com validação aninhada, o melhor erro médio absoluto (MAE) caiu de 1,130 s (Gradient Boosting, configuração padrão) para 1,008 s (ElasticNet), valor próximo ao ruído do piloto (desvio-padrão de 1,26 s). Conclui-se que modelos lineares regularizados são adequados ao volume de dados e que, em Mônaco, o piloto e o desgaste dos pneus limitam o poder preditivo do acerto.

**Palavras-chave:** aprendizado de máquina; telemetria; Fórmula 1; acerto de carro; validação cruzada agrupada; regressão regularizada.

---

## 1 Introdução

Mônaco é um caso extremo no calendário da Fórmula 1. A pista de 3,337 km é estreita, tem curvas de baixíssima velocidade (o Grampo Loews é contornado a cerca de 50 km/h) e uma única reta digna do nome, o trecho do túnel. Nessas condições, o acerto do carro privilegia tração, estabilidade em frenagem e resposta nas curvas lentas, e não a velocidade final. Em uma equipe real, essas escolhas são orientadas por engenheiros de corrida que cruzam telemetria, simulação e a percepção do piloto.

Os simuladores modernos oferecem um laboratório acessível para estudar esse problema. O jogo *F1 23* (Codemasters/EA) transmite, em tempo real, pacotes UDP com posição, velocidade, uso de pedais, marchas, acerto do carro e histórico de voltas. Com isso, é possível montar um experimento controlado: mesma pista, mesmo piloto, mesmas condições climáticas, variando apenas o acerto.

Este trabalho propõe um "engenheiro de corrida virtual", isto é, um conjunto de modelos de aprendizado de máquina que, a partir do acerto e do contexto da volta, estima o tempo de volta, a velocidade máxima e a velocidade média em Mônaco. As perguntas de pesquisa são:

1. **(Q1)** Quais variáveis de acerto e de contexto mais se associam ao tempo de volta em Mônaco?
2. **(Q2)** Alterações isoladas de um fator (S1–S9) e um acerto multifatorial sugerido por IA (S10) produzem diferenças de tempo superiores ao ruído do próprio piloto?
3. **(Q3)** Qual família de modelos generaliza melhor para sessões nunca vistas, com um volume pequeno de dados (≈ 270 voltas)?

A principal contribuição é metodológica: mostrar, com dados reais de captura, como a validação agrupada por sessão e a comparação explícita com o ruído do piloto evitam conclusões otimistas comuns em projetos com poucos dados.

## 2 Metodologia

### 2.1 Coleta de dados: telemetria UDP do F1 23

A captura foi feita com o programa `capturar_telemetria.py`, escrito em Python usando apenas a biblioteca padrão. Ele escuta a porta UDP 20777 e decodifica os pacotes no formato 2023 descrito na especificação oficial "Data Output from F1 23" (EA/Codemasters). Foram usados os pacotes *Motion* (posição `world_x/world_z` e forças G), *Session* (pista; `trackId = 5` corresponde a Mônaco), *Lap Data*, *Car Telemetry* (velocidade, acelerador, freio, marcha, DRS), *Car Setups* e *Session History*. O programa grava dois arquivos:

- `voltas.csv`: uma linha por volta completada, com 81 colunas (tempos de volta e de setor, acerto, idade do pneu, combustível, flags de validade e agregados de telemetria);
- `amostras.csv`: uma linha por instante de telemetria (≈ 20 Hz), com 1.543.167 amostras distribuídas em 57 sessões de captura.

Todas as voltas foram feitas em Mônaco, em pista seca, no modo Grande Prêmio › Treino Livre, com o desempenho real dos carros habilitado. O modo *Time Trial* foi descartado porque iguala o desempenho dos carros. As capturas ocorreram entre 13 e 22 de outubro de 2026.

### 2.2 Desenho experimental

Foram usados dois carros: Ferrari (equipe 1) e Alfa Romeo (equipe 9). O acerto base S0 foi adaptado de um acerto público de *Time Trial* para Mônaco (TimeTrialKing, simracingsetup.com). Os acertos S1 a S9 alteram **um único fator** em relação ao S0, para que a diferença de tempo tenha uma causa identificável. O S10 foi sugerido por uma IA generativa e altera 12 parâmetros, agrupados em 8 famílias, ao mesmo tempo. Ele funciona como o teste "IA × engenheiro". A Tabela 1 resume os valores efetivamente registrados no `voltas.csv`.¹

**Tabela 1 — Acertos testados (valores registrados na telemetria; campos em branco = igual ao S0)**

| Setup | Parâmetro alterado | Valor no S0 | Valor no setup |
|---|---|---|---|
| S0 | Base: asa 50/50; dif. 50/53; cambagem −2,5/−1,0; conv. 0,05/0,25; susp. 41/13; barras 7/5; altura 32/36; freio 53 %; pneus 22,5/20,0 psi | — | — |
| S1 | Diferencial com aceleração (e sem aceleração) | 50 (53) | 80 (50) |
| S2 | Diferencial sem aceleração | 53 | 80 |
| S3 | Cambagem dianteira/traseira | −2,5/−1,0 | −3,5/−2,0 |
| S4 | Convergência traseira | 0,25 | 0,10 |
| S5 | Barra estabilizadora dianteira | 7 | 15 |
| S6 | Barra estabilizadora traseira | 5 | 12 |
| S7 | Altura dianteira/traseira | 32/36 | 30/45 |
| S8 | Pressão do pneu traseiro (psi) | 20,0 | 22,0 |
| S9 | Asa traseira | 50 | 45 |
| S10 (IA) | Asa T 47; dif. sem acel. 50; conv. 0,08/0,20; susp. 12/7; barras 5/2; altura 36/42; freio 54 %; pneu D 22,2 psi | — | multifatorial |

O S0 foi rodado no início e no fim das sessões para medir a variabilidade do piloto. O protocolo previa duas voltas de aquecimento e cerca de dez voltas cronometradas por acerto. Na prática, cada combinação carro × acerto teve entre 9 e 34 voltas (o S0 da Ferrari concentra 34).

### 2.3 Conjunto de dados

O `voltas.csv` final contém **277 voltas**: 144 da Ferrari, 129 da Alfa Romeo e 4 de uma McLaren, todas apenas no S0. As voltas vêm de 40 sessões distintas (`session_uid`): 17 da Ferrari, 22 da Alfa Romeo e 1 da McLaren. As 4 voltas da McLaren foram excluídas da modelagem por serem poucas demais, restando **273 voltas**. Doze colunas são constantes (por exemplo, pista, modo de jogo e lastro) e duas estão inteiramente vazias (`desempenho_igual` e `freio_motor`). Por isso, todas foram descartadas.

Na análise de qualidade, 42 voltas (15,2 %) aparecem marcadas como volta de saída, 25 (9,0 %) terminam no box e 100 registram ao menos um *flashback*. Em vez de descartar essas voltas, o que reduziria ainda mais uma amostra já pequena, a modelagem as manteve e incluiu indicadores binários e contagens (`volta_saida`, `entrou_box`, `flashbacks`) como variáveis de contexto.² O corte de 107 % da mediana (limite de 83,692 s) afetaria apenas uma volta e não foi aplicado na modelagem.

### 2.4 Variáveis

**Alvos.** Foram modelados três alvos independentes: (i) tempo de volta (s); (ii) velocidade máxima na volta (km/h); e (iii) velocidade média na volta (km/h).

**Preditores (24).** Foram usados 18 parâmetros de acerto (asas, diferenciais, cambagem, convergência, suspensão, barras, alturas, balanço de freio, pressões de pneu dianteiro e traseiro e combustível do acerto), 4 variáveis de contexto (idade do pneu em voltas, volta de saída, entrada no box e número de *flashbacks*) e 2 indicadores (carro = Ferrari; composto visual = duro). **Tempos de setor, velocidades em curvas e demais agregados de telemetria não foram usados como preditores** dos modelos de acerto, porque são consequências da volta, e não decisões tomadas antes dela. Usá-los vazaria informação do alvo. Esses agregados aparecem apenas em um modelo *explicativo* separado (Seção 3.4).

### 2.5 Modelos e validação

Foram comparados: a média do treino (*baseline*), uma regressão linear apenas com o carro, Lasso, Ridge, ElasticNet, Random Forest (RF), Extra Trees (ET) e Gradient Boosting (GBM), todos implementados em scikit-learn. Os modelos lineares receberam padronização (`StandardScaler`) em *pipeline*. Uma rede neural *Multilayer Perceptron* (MLP), com alvo padronizado, foi avaliada apenas no tempo de volta, como experimento opcional de *deep learning*.

Como voltas da mesma sessão compartilham condições (pneus, combustível, ritmo do piloto naquele dia), uma validação cruzada aleatória superestimaria o desempenho. Por isso, adotou-se:

- **GroupKFold com 5 blocos agrupados por `session_uid`**: nenhuma sessão aparece ao mesmo tempo em treino e validação;
- **teste *hold-out* com 20 % das sessões** (GroupShuffleSplit, semente 42), com 57 voltas de sessões nunca vistas;
- **ajuste de hiperparâmetros** com RandomizedSearchCV (30 combinações, GroupKFold interno de 4 blocos, otimizando o MAE) em **validação cruzada aninhada**. A busca ocorre apenas dentro do treino de cada bloco externo, o que evita estimativas otimistas.

As métricas reportadas são MAE, RMSE e R². A importância das variáveis foi estimada por seis métodos: |coeficiente| do Lasso, importância por impureza (RF e GBM) e importância por permutação nos blocos de validação (RF, Lasso e GBM). Os seis rankings foram combinados em uma média normalizada (consenso). As comparações entre acertos usaram o teste de Mann-Whitney (α = 5 %), e as associações monotônicas, a correlação de Spearman. O projeto também mantém notebooks equivalentes para o Google Colab e um leitor de dados em R, na linha do fluxo *tidymodels* trabalhado na disciplina. Os modelos finais foram publicados em uma API (FastAPI) consumida pela página do projeto.

## 3 Resultados

### 3.1 Análise exploratória

O tempo médio de volta foi de 78,093 s na Ferrari (DP 1,52 s) e de 78,658 s na Alfa Romeo (DP 1,51 s). A diferença de 0,565 s é significativa (Mann-Whitney, p = 0,0039) (Figura 1). A **volta mais rápida do conjunto foi 1:14.575 (74,575 s), da Ferrari com o S10**. Na Alfa Romeo, a melhor volta absoluta foi 75,852 s, no S8.

![Figura 1 — Distribuição do tempo de volta por carro.](figuras/01_dist_carros.png)

*Figura 1 — Distribuição do tempo de volta por carro (histograma e boxplot). A McLaren (4 voltas, S0) aparece apenas para referência.*

A telemetria confirma o perfil da pista. A maior velocidade registrada foi **289 km/h (Ferrari, S0), a 1.976 m da linha de chegada, na região do túnel**. O trecho mais lento fica em torno de 1.250 m (média de 50 km/h, Grampo Loews). A marcha mais usada é a 2ª (25,5 % do tempo em movimento), e a 8ª aparece em apenas 0,4 %. O setor 2, que concentra cerca de 48 % do tempo da volta, é o mais lento (133,7 km/h de média) e o que mais se correlaciona com o tempo total (r = 0,86). A correlação do tempo de volta é mais forte com a velocidade média (ρ = −0,46) do que com a velocidade máxima (ρ = −0,29). Em Mônaco, portanto, a velocidade nas curvas pesa mais que a velocidade final (Figuras 2 e 3).

![Figura 2 — Mapa de calor de velocidade por posição em Mônaco.](figuras/14_heatmap.png)

*Figura 2 — Mapa de Mônaco reconstruído a partir das coordenadas `world_x × world_z`, colorido pela velocidade média. Os pontos brancos marcam as zonas de frenagem da volta mais rápida.*

![Figura 3 — Traço de velocidade: volta mais rápida × mais lenta.](figuras/13_traco.png)

*Figura 3 — Velocidade ao longo da volta: a mais rápida (74,575 s, Ferrari S10) contra a mais lenta sem volta de saída (83,795 s, Alfa Romeo S2). O maior ganho ocorre por volta de 850–875 m (+51 km/h).*

**Pneus e combustível.** A idade do pneu se associa positivamente ao tempo (ρ = 0,45; p = 2,8 × 10⁻¹⁵). Sem as voltas de saída, a média passa de 77,346 s na 2ª volta do pneu para 79,206 s na 8ª. O combustível no início da volta tem associação fraca e não significativa (ρ = −0,12; p = 0,060). O consumo médio foi de 1,124 kg por volta (Figura 4).

![Figura 4 — Efeito do stint e do combustível.](figuras/05_stint.png)

*Figura 4 — Tempo médio por idade do pneu (esquerda) e combustível no início da volta × tempo (direita).*

**Ruído do piloto.** Nas voltas do S0, o desvio-padrão foi de 1,23 s na Ferrari (34 voltas) e de 1,38 s na Alfa Romeo (11 voltas), com DP combinado de **1,26 s** (Figura 5). Esse valor serve de régua: uma diferença de acerto menor que ela não se distingue da variação natural entre voltas do mesmo piloto.³

![Figura 5 — Ruído do piloto no S0.](figuras/06_ruido_s0.png)

*Figura 5 — Dispersão das voltas no acerto base S0 por carro, usada como estimativa do ruído do piloto.*

### 3.2 Efeito dos acertos

A Tabela 2 e as Figuras 6 e 7 resumem o efeito de cada acerto em relação ao S0 do mesmo carro.

**Tabela 2 — Tempo médio por acerto (s) e diferença em relação ao S0**

| Setup | Ferrari: média (n) | Δ vs S0 | Alfa Romeo: média (n) | Δ vs S0 |
|---|---|---|---|---|
| S0 | 78,303 (34) | — | 79,151 (11) | — |
| S1 | 77,876 (14) | −0,43 | 77,925 (11) | −1,23 |
| S2 | 79,397 (11) | **+1,09** | 81,078 (12) | **+1,93** |
| S3 | 77,943 (11) | −0,36 | 78,055 (11) | −1,10 |
| S4 | 78,880 (12) | +0,58 | 78,612 (11) | −0,54 |
| S5 | 77,435 (12) | −0,87 | 78,774 (13) | −0,38 |
| S6 | 78,176 (11) | −0,13 | 78,151 (11) | −1,00 |
| S7 | 77,779 (9) | −0,52 | 79,110 (12) | −0,04 |
| S8 | 78,764 (9) | +0,46 | 78,576 (12) | −0,58 |
| S9 | 77,366 (9) | −0,94 | 78,132 (13) | −1,02 |
| S10 (IA) | **76,767** (12) | **−1,54** | **77,596** (12) | **−1,55** |

*Médias calculadas a partir do `voltas.csv` (273 voltas de Ferrari e Alfa Romeo). Os Δ foram arredondados para duas casas.*

![Figura 6 — Tempo de volta por setup e carro.](figuras/02_box_setups.png)

*Figura 6 — Distribuição do tempo de volta por acerto e carro.*

![Figura 7 — Delta médio de cada setup em relação ao S0.](figuras/03_delta_setups.png)

*Figura 7 — Diferença média de cada acerto em relação ao S0 do próprio carro (valores negativos indicam voltas mais rápidas).*

O **S10, sugerido por IA, foi o melhor acerto nos dois carros**: −1,54 s na Ferrari e −1,55 s na Alfa Romeo. Ele também foi o acerto mais consistente da Alfa Romeo (DP 0,87 s). O **S2 (diferencial sem aceleração em 80) foi o pior nos dois carros**: +1,93 s na Alfa Romeo e +1,09 s na Ferrari. Na Alfa Romeo, a perda do S2 se concentra no setor 2 (+1,50 s, contra −0,03 s no S1 e +0,47 s no S3; Figura 8). A telemetria localiza a maior perda de velocidade em torno de 1.200 m (−12,7 km/h, região do Grampo). Já o S10 ganha em média 3,0 km/h, com pico de +20,6 km/h por volta de 950 m (Figura 9). Na Ferrari, o S10 ganhou em todos os setores (S1 −0,41 s; S2 −0,87 s; S3 −0,26 s).

![Figura 8 — Delta por setor na Alfa Romeo.](figuras/04_delta_setor_alfa.png)

*Figura 8 — Alfa Romeo: diferença média por setor de cada acerto em relação ao S0.*

![Figura 9 — Alfa Romeo: S10 e S2 contra S0 ao longo da volta.](figuras/17_alfa_s10_s2.png)

*Figura 9 — Alfa Romeo: diferença de velocidade média por posição na volta (S10 − S0 e S2 − S0).*

Apesar dessas tendências, **apenas 3 das 20 comparações contra o S0 foram estatisticamente significativas** (Mann-Whitney, α = 5 %): Alfa Romeo S2, Alfa Romeo S10 e Ferrari S10. São também as únicas diferenças maiores que o ruído do piloto (1,26 s). Dos 10 acertos alternativos, 9 melhoraram a média da Alfa Romeo e 7 a da Ferrari. A concordância de ranking entre os carros foi positiva, mas não significativa (Spearman ρ = 0,49; p = 0,150).

### 3.3 Importância das variáveis

Os seis métodos de importância apontaram o mesmo trio no topo (Figura 10). No consenso normalizado, a **idade do pneu** respondeu por **42,7 %** da importância, o **diferencial sem aceleração** (*off-throttle*) por **26,0 %** e o **carro (Ferrari)** por **15,0 %**. Em seguida vêm a convergência dianteira (3,4 %) e o combustível do acerto (2,2 %). A idade do pneu ocupou a mesma posição em todos os métodos. A convergência dianteira, por outro lado, variou 17 posições: o Lasso a valorizou e os modelos de árvore praticamente a ignoraram.

O Lasso (LassoCV) zerou 11 das 24 variáveis. Os maiores coeficientes, em segundos por desvio-padrão, foram: idade do pneu +0,667, diferencial sem aceleração +0,540, carro Ferrari −0,420 e convergência dianteira −0,252. Os sinais são coerentes com a análise exploratória: pneu mais gasto e diferencial mais travado na desaceleração pioram o tempo, e a Ferrari é mais rápida.

![Figura 10 — Importância consensual das variáveis.](figuras/09_consenso.png)

*Figura 10 — Importância média normalizada (%) das 12 principais variáveis, combinando seis métodos (|coef.| Lasso, impureza RF/GBM, permutação RF/Lasso/GBM).*

### 3.4 Desempenho preditivo — tempo de volta

A Tabela 3 apresenta os resultados da modelagem principal, com hiperparâmetros fixos ou ajustados internamente (LassoCV/RidgeCV).

**Tabela 3 — Tempo de volta: métricas da validação cruzada agrupada (CV) e do teste *hold-out***

| Modelo | MAE CV (s) | RMSE CV (s) | R² CV | MAE teste (s) | R² teste |
|---|---|---|---|---|---|
| Baseline (média) | 1,219 | 1,557 | −0,03 | 1,164 | −0,08 |
| Só carro (linear) | 1,226 | 1,550 | −0,02 | 1,179 | −0,11 |
| **Lasso (LassoCV)** | **1,015** | **1,274** | **0,31** | **1,060** | **0,19** |
| Ridge (RidgeCV) | 1,125 | 1,454 | 0,10 | 1,105 | 0,15 |
| Random Forest | 1,062 | 1,360 | 0,22 | 1,238 | −0,04 |
| Gradient Boosting | 1,089 | 1,354 | 0,22 | 1,141 | 0,06 |
| Extra Trees | 1,076 | 1,375 | 0,20 | 1,282 | −0,11 |

![Figura 11 — Comparação de modelos para o tempo de volta.](figuras/11_modelos.png)

*Figura 11 — MAE e R² na validação cruzada e no teste para o tempo de volta.*

Para isolar o efeito do ajuste de hiperparâmetros, todos os algoritmos foram comparados em duas versões: a configuração padrão do scikit-learn e a versão ajustada com busca aleatória em validação aninhada (Tabela 4, Figura 12). **Antes do ajuste, o melhor modelo foi o Gradient Boosting, com MAE CV de 1,130 s**.⁴ **Depois do ajuste, os lineares regularizados assumiram a liderança: ElasticNet com 1,008 s (R² CV 0,33) e Lasso com 1,017 s (R² CV 0,32)**. A diferença entre os dois (0,009 s) é desprezível.

**Tabela 4 — Tempo de volta: MAE antes e depois do ajuste de hiperparâmetros (CV aninhada)**

| Modelo | MAE CV padrão | MAE CV ajustado | Ganho | MAE teste padrão | MAE teste ajustado |
|---|---|---|---|---|---|
| **ElasticNet** | 1,184 | **1,008** | −14,9 % | 1,093 | 1,049 |
| Lasso | 1,219⁵ | 1,017 | −16,5 % | 1,164 | 1,086 |
| Gradient Boosting | 1,130 | 1,049 | −7,1 % | 1,158 | 1,075 |
| Extra Trees | 1,239 | 1,094 | −11,7 % | 1,184 | 1,024 |
| Random Forest | 1,153 | 1,097 | −4,8 % | 1,234 | 1,184 |
| Ridge | 1,179 | 1,143 | −3,1 % | 1,144 | 1,075 |
| MLP (rede neural) | 1,147 | 1,365 | +19,0 % (piora) | 1,135 | 0,977 |
| Baseline (média) | 1,219 | — | — | 1,164 | — |

![Figura 12 — MAE antes e depois do ajuste.](figuras/18_tuning_mae.png)

*Figura 12 — MAE da validação cruzada agrupada antes (cinza) e depois (vermelho) do ajuste de hiperparâmetros. A linha tracejada indica o baseline (1,219 s). Gráfico gerado a partir de `resumo_mae.csv`.*

Os melhores hiperparâmetros encontrados foram: Lasso α = 0,140; Ridge α = 41,4; ElasticNet α = 0,411 e l1_ratio = 0,55; RF com 465 árvores, profundidade máxima 5, folha mínima 9 e max_features 0,5; ET com 328 árvores, folha mínima 3 e max_features 0,3; GBM com 402 árvores de profundidade 1 (*stumps*), taxa de aprendizado 0,011 e subamostragem 0,92; MLP com camadas (32, 16) e α = 0,018. Nos modelos de árvore, a busca escolheu sistematicamente estruturas **mais simples**, o que é coerente com o tamanho da amostra. A MLP piorou na CV depois do ajuste (1,365 s), mas teve o menor MAE no *hold-out* (0,977 s). Como o *hold-out* tem apenas ≈ 55–57 voltas, esse resultado foi interpretado como instabilidade, e não como superioridade. A Figura 13 mostra a dispersão previsto × real no teste para o Lasso.

![Figura 13 — Previsto × real no teste (Lasso).](figuras/12_pred_real.png)

*Figura 13 — Tempo previsto × tempo real nas sessões de teste (Lasso). A linha tracejada indica a previsão perfeita.*

**Modelo explicativo de telemetria.** Como contraponto, um RF treinado apenas com agregados de telemetria (velocidades em curvas, aceleração lateral média, % de acelerador e de freio) atingiu MAE CV de 0,820 s e R² de 0,52. As variáveis mais relevantes foram a aceleração lateral média, a velocidade máxima no túnel e a % do tempo freando (Figura 14). Esse modelo não serve para prever o efeito de um acerto antes da volta, mas mostra que a informação "faltante" no modelo de acerto está na execução da volta.

![Figura 14 — Importância no modelo explicativo de telemetria.](figuras/10_imp_tel.png)

*Figura 14 — Importância por permutação no modelo explicativo de telemetria (não usado para previsão de acerto).*

### 3.5 Desempenho preditivo — velocidades máxima e média

**Tabela 5 — Velocidades: melhor modelo × baseline (MAE em km/h)**

| Alvo | Melhor modelo | MAE CV | R² CV | MAE teste | Baseline MAE CV | Baseline MAE teste |
|---|---|---|---|---|---|---|
| Velocidade máxima | Random Forest | 5,16 | 0,10 | 4,54 | 5,72 | 4,84 |
| Velocidade média | Lasso | 3,99 | 0,59 | 3,09 (R² 0,77) | 7,10 | 6,83 |

Para a **velocidade máxima**, o melhor modelo (RF, MAE CV ≈ 5,2 km/h) supera o baseline (5,7 km/h) por margem pequena, com R² CV de apenas 0,10. Os lineares regularizados praticamente reproduziram a média. Para a **velocidade média**, o Lasso reduziu o erro de ≈ 7,1 para ≈ 4,0 km/h (R² CV 0,59), e o RF ficou próximo (4,08 km/h) (Figura 15).⁶

![Figura 15 — MAE por alvo e modelo.](figuras/19_mae_alvos.png)

*Figura 15 — MAE da validação cruzada agrupada para os três alvos. Barras cinza indicam o baseline (média). Gráfico gerado a partir de `metricas_modelos.csv`.*

## 4 Discussão

**Q1 — Variáveis relevantes.** O desgaste do pneu foi o fator dominante, à frente de qualquer ajuste de acerto, o que é coerente com uma pista de tração em que a borracha se degrada rapidamente. Entre os parâmetros de acerto, o diferencial sem aceleração se destacou de forma consistente. Fisicamente, um diferencial muito travado na desaceleração dificulta a rotação do carro na entrada das curvas lentas, como o Grampo, a Rascasse e a Sainte-Dévote. O resultado do S2 e a perda localizada no setor 2 e em torno de 1.200 m corroboram essa interpretação. A importância do carro (Ferrari ≈ 0,57 s mais rápida em média) confirma que o modo com desempenho real dos carros captou diferenças entre equipes.

**Q2 — Acertos versus ruído.** O ponto central do estudo é a comparação com o ruído do piloto. Com DP de 1,26 s entre voltas do mesmo acerto, apenas o S10 (nos dois carros) e o S2 (na Alfa Romeo) produziram efeitos claramente distinguíveis. O S10 multifatorial, sugerido por IA, ganhou cerca de 1,5 s, e os ajustes isolados ficaram, em sua maioria, dentro do ruído. Isso sugere que os ganhos em Mônaco vêm da combinação de ajustes, e não de um único "clique" mágico. O desenho experimental, porém, não permite atribuir o ganho do S10 a parâmetros específicos, porque doze valores mudaram ao mesmo tempo.

**Q3 — Escolha de modelo.** Com 273 voltas e 24 preditores, os modelos lineares regularizados (ElasticNet e Lasso) generalizaram melhor que os *ensembles* de árvore e que a MLP. O ajuste de hiperparâmetros melhorou todos os modelos clássicos entre 3 % e 17 %, mas o ganho absoluto foi de apenas 0,04–0,20 s. O melhor MAE (≈ 1,0 s) supera o baseline em cerca de 0,2 s e fica **abaixo** do ruído do piloto, mas o R² máximo de ≈ 0,33 indica que a maior parte da variância não é explicada pelo acerto. A boa performance do modelo explicativo de telemetria (MAE 0,82 s) reforça que essa variância restante está na execução da volta: traçado, pontos de frenagem e velocidade em curva.

A rede neural foi incluída como exploração de *deep learning*. Seu comportamento instável, com piora na CV após o ajuste e melhora no *hold-out*, é um exemplo didático de como um *hold-out* pequeno pode enganar. Também mostra por que a validação agrupada e aninhada deve ser o critério de seleção.

**Ameaças à validade.**
- *Confundimento por combustível*: o combustível do acerto variou entre 5 e 20 kg. S4, S7 e S10 rodaram com 5 kg na Alfa Romeo, contra 13 kg no S0. Parte do ganho atribuído ao acerto pode vir do carro mais leve.
- *Bases diferentes de S0*: o S0 da Ferrari inclui voltas com valores distintos do S0 nominal (asa 45/45, balanço de freio 62 %, 20 kg de combustível). Assim, os Δ dos dois carros partem de bases ligeiramente diferentes.
- *Composto*: o pneu duro só aparece na Ferrari (S0, S1 e S6), o que confunde composto e acerto.
- *Aprendizado do piloto*: os tempos médios melhoram entre sessões (por exemplo, Ferrari 78,51 → 77,79 → 77,60 s), então a ordem das sessões interfere no efeito do acerto.
- *Qualidade da captura*: a marcação de volta de saída parece imprecisa, já que essas voltas não foram mais lentas (−0,80 s), e houve picos fisicamente impossíveis de aceleração lateral (48,6 g), que exigem limpeza.
- *Generalização*: um único piloto, uma única pista e um simulador limitam a extrapolação para a Fórmula 1 real.

## 5 Conclusão

Este trabalho construiu, de ponta a ponta, um "engenheiro de corrida virtual" para Mônaco no *F1 23*: captura de telemetria UDP em Python, experimento controlado com onze acertos e dois carros, modelagem com validação agrupada por sessão e publicação dos modelos em uma API. Os principais achados foram:

1. a idade do pneu (≈ 43 %), o diferencial sem aceleração (≈ 26 %) e o carro (≈ 15 %) concentram a importância preditiva do tempo de volta;
2. o acerto multifatorial S10, sugerido por IA, reduziu o tempo médio em ≈ 1,5 s nos dois carros e produziu a volta mais rápida do conjunto (1:14.575, Ferrari), enquanto o S2 foi o pior acerto (+1,93 s na Alfa Romeo);
3. apenas 3 de 20 comparações de acerto superaram o ruído do piloto (1,26 s);
4. após ajuste com validação aninhada, o ElasticNet (MAE 1,008 s) e o Lasso (1,017 s) superaram o melhor modelo padrão (GBM, 1,130 s), com o piloto como limite prático do erro;
5. a velocidade média é bem prevista (Lasso, ≈ 4 km/h contra ≈ 7 km/h do baseline), mas a velocidade máxima quase não é (RF ≈ 5,2 km/h contra 5,7 km/h).

Como trabalhos futuros, propõem-se: fixar o combustível e o composto entre acertos; randomizar a ordem dos acertos para neutralizar o aprendizado do piloto; aumentar o número de voltas por acerto, a fim de ganhar poder estatístico; usar planejamento fatorial fracionado para decompor o efeito do S10; e incorporar variáveis de execução (por exemplo, velocidade mínima nas curvas) em um modelo em dois estágios, com o acerto prevendo a execução e a execução prevendo o tempo.

## Notas

¹ O documento de planejamento (`setups_f123.md`) descreve o S2 como diferencial sem aceleração em 30 e o S7 como altura 25/45. Os valores efetivamente registrados pela telemetria são 80 e 30/45, respectivamente, e foram os adotados neste artigo. O S1 registrado também altera o diferencial sem aceleração de 53 para 50, além de mudar o diferencial com aceleração para 80.

² A página do projeto menciona a remoção de voltas de saída, de box e com *flashback*. Os scripts de modelagem (`analise.py` e `tuning.py`), porém, mantêm as 273 voltas e tratam essas condições como variáveis. Este artigo descreve o procedimento dos scripts.

³ Um valor de 1,52 s citado anteriormente como "ruído do piloto" corresponde ao DP de **todas** as voltas da Ferrari, que mistura piloto e acerto. A medida restrita ao S0 é 1,26 s.

⁴ A análise principal (Tabela 3) usou GBM com hiperparâmetros escolhidos manualmente (300 árvores, profundidade 2), obtendo 1,089 s. O valor de 1,130 s refere-se ao GBM com a configuração padrão do scikit-learn no experimento de ajuste (Tabela 4). Excluindo o Lasso padrão, que coincide com o baseline (nota 5), o GBM é o melhor modelo nessa condição.

⁵ Com α = 1 (padrão do scikit-learn), o Lasso zera todos os coeficientes e reproduz o baseline.

⁶ Não foi feita uma análise de importância de variáveis específica para os alvos de velocidade. É plausível que parte do desempenho na velocidade média venha dos indicadores `entrou_box` e `volta_saida`, mas essa hipótese não foi testada nos dados disponíveis. Também não há, nos arquivos do projeto, um ajuste de hiperparâmetros para esses dois alvos; os valores da Tabela 5 vêm da modelagem principal. Um resultado de "RF 1,03 s no teste", citado em materiais anteriores, vem de outra partição treino/teste e não é reproduzível a partir dos arquivos atuais, por isso não foi usado.

## Referências

BERGSTRA, J.; BENGIO, Y. Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, v. 13, p. 281–305, 2012.

BREIMAN, L. Random forests. *Machine Learning*, v. 45, n. 1, p. 5–32, 2001.

ELECTRONIC ARTS; CODEMASTERS. *Data Output from F1 23* (especificação do protocolo UDP de telemetria do jogo F1 23). Documentação oficial publicada nos fóruns da EA; cópia de referência em: <https://github.com/MacManley/f1-23-udp>. Acesso em: out. 2026.

FRIEDMAN, J. H. Greedy function approximation: a gradient boosting machine. *The Annals of Statistics*, v. 29, n. 5, p. 1189–1232, 2001.

GEURTS, P.; ERNST, D.; WEHENKEL, L. Extremely randomized trees. *Machine Learning*, v. 63, n. 1, p. 3–42, 2006.

HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning: data mining, inference, and prediction*. 2. ed. New York: Springer, 2009.

HOERL, A. E.; KENNARD, R. W. Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, v. 12, n. 1, p. 55–67, 1970.

KUHN, M.; SILGE, J. *Tidy Modeling with R*. Sebastopol: O'Reilly, 2022. Disponível em: <https://www.tmwr.org/>.

MANN, H. B.; WHITNEY, D. R. On a test of whether one of two random variables is stochastically larger than the other. *The Annals of Mathematical Statistics*, v. 18, n. 1, p. 50–60, 1947.

MARCELINO, A. *f1-ml-monaco*: Engenheiro de Corrida Virtual — F1 23 Mônaco (código, dados e modelos). GitHub, 2026. Disponível em: <https://github.com/andremarcelino-mp4/f1-ml-monaco>.

MARCELINO, A. *Engenheiro de Corrida Virtual · F1 23 Mônaco* (página do projeto e simulador interativo). 2026. Disponível em: <https://f1-monaco-landing.vercel.app/>.

PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825–2830, 2011.

SIMRACINGSETUP. *F1 23 Monaco setup* (acerto TimeTrialKing usado como base S0). Disponível em: <https://simracingsetup.com/>. Acesso em: out. 2026.

TIBSHIRANI, R. Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, v. 58, n. 1, p. 267–288, 1996.

VARMA, S.; SIMON, R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, v. 7, art. 91, 2006.

ZOU, H.; HASTIE, T. Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, v. 67, n. 2, p. 301–320, 2005.
