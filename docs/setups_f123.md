# Setups F1 23 – Mônaco (Ferrari e Alfa Romeo)

Base S0: TimeTrialKing, Time Trial, seco, Equal performance, 1:08.346 (simracingsetup.com, F1 23).

| Parâmetro | S0 |
|---|---|
| Asa D/T | 50/50 |
| Dif. com/sem acel. | 50/53 |
| Cambagem D/T | -2,5/-1,0 |
| Convergência D/T | 0,05/0,25 |
| Suspensão D/T | 41/13 |
| Barras D/T | 7/5 |
| Altura D/T | 32/36 |
| Freio pressão/balanço | 100/53 |
| Pneus D/T | 22,5/20,0 psi |

## Sessões (só muda o que está em negrito; o resto = S0)

| Setup | Dif. | Cambagem | Conv. T | Barras | Altura | Pneu T | Asa T | Testa |
|---|---|---|---|---|---|---|---|---|
| S0 | 50/53 | -2,5/-1,0 | 0,25 | 7/5 | 32/36 | 20,0 | 50 | Base |
| S1 | **80**/53 | | | | | | | Dif. travado |
| S2 | 50/**30** | | | | | | | Menos trava entrada |
| S3 | | **-3,5/-2,0** | | | | | | Cambagem agressiva |
| S4 | | | **0,10** | | | | | Traseira solta |
| S5 | | | | **15**/5 | | | | Barra D dura |
| S6 | | | | 7/**12** | | | | Barra T dura |
| S7 | | | | | **25/45** | | | Altura invertida |
| S8 | | | | | | **22,0** | | Pneu T cheio |
| S9 | | | | | | | **45** | Controle menos asa |

Rotina: 2 carros (Ferrari equipe 1, Alfa Romeo equipe 9) × 10 setups × (2 aquecimento + 10 válidas). S0 no início e fim de cada sessão. ≥2 sessões por carro (ex.: S0–S4 e S0,S5–S9).
Exemplo: `python capturar_telemetria.py --carro ferrari --setup S1 --sessao 1`
Obs.: Time Trial F1 23 = desempenho igual; para carros diferentes use Grande Prêmio > Treino Livre.
