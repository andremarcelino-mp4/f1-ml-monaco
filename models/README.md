# Modelos

Cada pasta tem `modelo.joblib` (pipeline do scikit-learn) e `meta.json` (algoritmo, hiperparâmetros, métricas e a lista de colunas na ordem de entrada). O `registro.json` lista todos e indica o modelo padrão de cada alvo.

| Alvo | Unidade | Pastas |
|---|---|---|
| tempo de volta | s | `tempo__*` |
| velocidade máxima | km/h | `vel_max__*` |
| velocidade média | km/h | `vel_media__*` |

Algoritmos de cada alvo: baseline (média), só carro, Lasso, Ridge, Random Forest, LightGBM e XGBoost. Padrões no registro: `tempo__random_forest`, `vel_max__lightgbm`, `vel_media__lasso`.

## Aviso importante

O `registro.json` marca `"sintetico": true`: estes `.joblib` foram gerados pelo pipeline da API e as métricas dentro dos `meta.json` (por exemplo, MAE de 0,55 s no Lasso de tempo) **não são** as métricas do estudo. Os números do estudo são os de `data/metricas_modelos.csv` e `data/hiperparametros/`, e o notebook em `notebooks/` retreina tudo a partir de `data/voltas.csv`. ElasticNet e Gradient Boosting tunados aparecem no tuning, mas não foram salvos como `.joblib`; para tê-los, rode o notebook.

## Uso

```python
import joblib, json, pandas as pd
m = joblib.load("models/tempo__lasso/modelo.joblib")
cols = json.load(open("models/tempo__lasso/meta.json", encoding="utf-8"))["colunas"]
# monte um DataFrame com essas colunas (18 do setup + carro + volta_stint) e chame m.predict
```

Salvos com scikit-learn 1.9.1.
