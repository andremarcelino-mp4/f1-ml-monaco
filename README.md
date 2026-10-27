# Engenheiro de corrida virtual: F1 23 em Mônaco

Este projeto nasceu de uma pergunta simples: dá para usar aprendizado de máquina para descobrir qual ajuste de carro deixa a volta mais rápida em Mônaco? Gravei a telemetria UDP do F1 23 enquanto pilotava, juntei as voltas num dataset e treinei modelos para prever tempo de volta, velocidade máxima e velocidade média a partir do setup. A ideia é que o modelo funcione como um engenheiro de pista: ele olha o acerto e diz o que esperar antes de você ir para a pista.

## O que eu encontrei

- **277 voltas válidas** em Mônaco, em 40 sessões: 144 de Ferrari, 129 de Alfa Romeo e 4 de McLaren (só no S0). Os modelos usam as 273 voltas de Ferrari e Alfa.
- **Melhor volta: 1:14.575**, com a Ferrari no setup S10. Nos dois carros, o S10 ficou cerca de 1,5 s mais rápido que o setup base (S0).
- **O que mais pesa no tempo:** a idade do pneu (cerca de 43% da importância média) e o diferencial sem aceleração (cerca de 26%). O carro vem logo depois.
- **Erro dos modelos de tempo de volta**, em validação cruzada por sessão (GroupKFold, 5 blocos):
  - ElasticNet tunado: MAE de **1,008 s**
  - Lasso tunado: 1,017 s
  - Gradient Boosting tunado: 1,049 s
  - Baseline (prever sempre a média): 1,219 s
- O tuning melhorou os modelos clássicos de 3% a 17%, mas o piso continua perto de 1 segundo. Esse piso é a minha própria variação como piloto: mesmo com o mesmo setup, as voltas oscilam cerca de 1,26 s. O R² máximo em CV ficou em torno de 0,33.
- Para **velocidade média**, o Lasso chegou a MAE de 3,09 km/h no teste (R² = 0,77). A **velocidade máxima** quase não é explicável pelo setup: o melhor modelo (Random Forest) erra 4,54 km/h, praticamente o mesmo que o baseline.

Os números vêm de `data/conclusoes.json`, `data/metricas_modelos.csv` e `data/hiperparametros/resumo_mae.csv`.

## Os dados

Cada linha de `data/voltas.csv` é uma volta: os 18 parâmetros do setup, o carro, o composto e a idade do pneu, flashbacks, tempos de setor e velocidades em pontos da pista (Sainte-Dévote, túnel, piscina etc.). Testei 11 setups (S0 a S10) em cada carro, voltando ao S0 no começo e no fim das sessões para medir o meu ruído. Detalhes das colunas e dos problemas conhecidos estão em [`data/README.md`](data/README.md).

## Como rodar o notebook

1. Abra o [Google Colab](https://colab.research.google.com/) e vá em **Arquivo → Fazer upload de notebook**.
2. Escolha `notebooks/F1_Monaco_Treino_Tuning_Colab.ipynb`.
3. Quando a célula de carga pedir, envie `data/voltas.csv` (ou aponte para o arquivo no Drive ou pela URL raw do GitHub).
4. **Ambiente de execução → Executar tudo.** No Colab gratuito leva de 5 a 10 minutos. Para um teste rápido, mude `N_ITER` para 5.

O notebook faz a análise exploratória, treina os modelos com e sem tuning, compara o MAE e gera os gráficos de previsto × real, resíduos e importância.

## Como carregar um modelo

```python
import joblib, json
import pandas as pd

pasta = "models/tempo__lasso"
modelo = joblib.load(f"{pasta}/modelo.joblib")
meta = json.load(open(f"{pasta}/meta.json", encoding="utf-8"))

s10 = dict(asa_dianteira=50, asa_traseira=47, dif_com_acel=50, dif_sem_acel=50,
           cambagem_diant=-2.5, cambagem_tras=-1.0, convergencia_diant=0.08,
           convergencia_tras=0.20, suspensao_diant=12, suspensao_tras=7,
           barra_diant=5, barra_tras=2, altura_diant=36, altura_tras=42,
           freio_pressao=100, freio_balanco=54, pneu_diant=22.2, pneu_tras=20.0)

linha = pd.DataFrame([{**s10, "carro": "ferrari", "volta_stint": 3}])[meta["colunas"]]
print(modelo.predict(linha))  # tempo de volta previsto, em segundos
```

Os modelos foram salvos com scikit-learn 1.9.1. Com outra versão o `joblib.load` pode reclamar; nesse caso, instale a mesma versão. Antes de usar, leia o aviso em [`models/README.md`](models/README.md).

## Estrutura

```
data/        voltas, métricas, importâncias, conclusões e resultados do tuning
models/      modelos .joblib por alvo, com meta.json e registro.json
notebooks/   notebook do Colab (treino + tuning)
docs/        setups usados no F1 23 e a comparação S0 × S10
artigo/      artigo científico (Markdown, PDF e figuras)
```

## Links

- Site do projeto: https://f1-monaco-landing.vercel.app/
- Repositório: https://github.com/andremarcelino-mp4/f1-ml-monaco. Este aqui é uma versão reorganizada, só com os arquivos que importam para reproduzir o estudo.

## Limitações

São 273 voltas de um único piloto, numa única pista, num jogo. O ruído do piloto domina o sinal, então diferenças de setup menores que cerca de 1 s ficam dentro do erro. Parte das voltas tem flashback ou é volta de saída; isso está documentado em `data/README.md`.

## Autor

André Marcelino · FECAF

Licença MIT. Veja [`LICENSE`](LICENSE).
