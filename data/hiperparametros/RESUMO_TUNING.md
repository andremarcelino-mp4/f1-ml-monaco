# Ajuste de hiperparâmetros: F1 23 Mônaco (tempo de volta)

**Dados:** `voltas.csv`, com 273 voltas de Ferrari e Alfa Romeo. As 4 voltas de McLaren ficaram de fora. São 24 features: 18 de setup, 4 de contexto, mais carro e composto.
**Validação:** GroupKFold com 5 blocos por sessão (`session_uid`). O hold-out usa 20% das sessões (GroupShuffleSplit, seed 42).
**Busca:** RandomizedSearchCV com 30 combinações e GroupKFold de 4 blocos por sessão, otimizando o MAE. A busca roda em **CV aninhada**: ela acontece só dentro do treino de cada fold externo, então o MAE tunado é honesto e não fica otimista.

| Modelo | MAE CV padrão (s) | MAE CV tunado (s) | Ganho | MAE teste padrão | MAE teste tunado |
|---|---|---|---|---|---|
| ElasticNet | 1,184 | **1,008** | −14,9% | 1,093 | 1,049 |
| Lasso | 1,219* | 1,017 | −16,5% | 1,164 | 1,086 |
| Gradient Boosting | 1,130 | 1,049 | −7,1% | 1,158 | 1,075 |
| Extra Trees | 1,239 | 1,094 | −11,7% | 1,184 | 1,024 |
| Random Forest | 1,153 | 1,097 | −4,8% | 1,234 | 1,184 |
| Ridge | 1,179 | 1,143 | −3,1% | 1,144 | 1,075 |
| MLP (rede neural) | 1,147 | 1,365 | **piorou** (+19%) | 1,135 | 0,977** |
| Baseline (média) | 1,219 | – | – | 1,164 | – |

\* O Lasso com o padrão do sklearn (alpha = 1) zera todas as features e fica igual ao baseline. No relatório anterior usamos LassoCV, que já ajusta o alpha internamente e deu 1,015 s. Ou seja, o Lasso do relatório já estava "tunado".
\** O hold-out tem só cerca de 55 voltas. A MLP foi muito bem nele, mas foi a pior na CV. Isso é instabilidade, não vitória.

## Conclusões
- **O tuning ajuda, mas pouco em segundos.** Todos os modelos clássicos melhoraram de 3% a 17%, ou seja, de 0,04 s a 0,20 s de MAE. O melhor resultado ficou em cerca de 1,0 s, praticamente igual ao Lasso que já tínhamos (1,015 s).
- **Os lineares regularizados seguem na frente:** ElasticNet com 1,008 s e Lasso com 1,017 s. A diferença entre os dois (0,009 s) é desprezível, então o Lasso continua sendo a escolha mais simples de explicar.
- **O piso é o ruído do piloto.** O R² máximo em CV é de cerca de 0,33. Nenhum hiperparâmetro remove a variação de cerca de 1,26 s entre voltas do mesmo setup.
- **Nos modelos de árvore, o tuning escolheu simplificar:** Random Forest com max_depth 5 e min_samples_leaf 9, Gradient Boosting com tocos de profundidade 1 e learning rate de 0,011. Com 273 voltas, modelos complexos decoram ruído.
- **A rede neural (MLP) não compensa** com esse volume de dados. Ela fica instável e, depois da busca, piora na CV.
- **A importância das features é estável.** Em 6 dos 7 modelos tunados, as 3 features do topo são as mesmas: **idade do pneu > diferencial sem aceleração ≈ carro**. Na MLP, idade do pneu e carro ficam no topo.

## Melhores hiperparâmetros (busca nas 273 voltas)
Lasso α = 0,140 · Ridge α = 41,4 · ElasticNet α = 0,411, l1_ratio = 0,55 · RF: 465 árvores, profundidade 5, folha mínima 9, max_features 0,5 · Extra Trees: 328 árvores, folha mínima 3, max_features 0,3 · GBM: 402 árvores, profundidade 1, learning rate 0,011, subsample 0,92 · MLP: camadas (32, 16), α = 0,018.

## O que falar em sala
1. "Ajustamos os hiperparâmetros de 7 algoritmos com busca aleatória e validação cruzada por sessão. Todos os modelos clássicos melhoraram, até 16,5% no erro, mas o melhor continua em cerca de 1 segundo."
2. "Mesmo tunados, Lasso e ElasticNet vencem. Com 273 voltas, modelos simples generalizam melhor, e a rede neural ficou instável."
3. "O ranking de importância não mudou depois do tuning: idade do pneu, diferencial e carro. Isso mostra que a conclusão é robusta e não depende do modelo."

Arquivos: `comparacao_padrao_vs_tunado.csv`, `resumo_mae.csv`, `melhores_parametros.json`, `importancia_permutacao_tunados.csv`, `tuning.py`, `log*.txt`.
