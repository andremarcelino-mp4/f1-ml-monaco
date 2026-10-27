# Notebook

`F1_Monaco_Treino_Tuning_Colab.ipynb` foi feito para rodar no Google Colab sem instalar nada além do que a primeira célula instala.

1. Faça upload do notebook no Colab.
2. Envie `data/voltas.csv` quando pedir (ou use Drive / URL raw).
3. Executar tudo. Leva de 5 a 10 minutos; `N_ITER = 5` deixa mais rápido.

O que ele faz: análise exploratória, separação treino/teste por sessão (20% das sessões no teste), validação cruzada com GroupKFold por `session_uid`, treino com parâmetros padrão e com RandomizedSearchCV, tabela de MAE, gráficos (previsto × real, resíduos, importância, setup × carro, idade do pneu, Vmax e Vmédia) e o experimento Ferrari S10 / Alfa S2. Os resultados vão para `/content/f1_monaco_out/`.

Por que GroupKFold: voltas da mesma sessão são muito parecidas. Se uma fosse para o treino e outra para a validação, o erro sairia otimista.
