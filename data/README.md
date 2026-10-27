# Dados

Toda a telemetria foi coletada por mim, pela saída UDP do F1 23 (formato 2023), pilotando em Mônaco. Não há dados de terceiros nem da Fórmula 1 oficial. F1 23 é marca da EA/Codemasters; aqui ficam só números gerados durante as minhas sessões, publicados sob a mesma licença MIT do repositório.

## Arquivos

| Arquivo | O que tem |
|---|---|
| `voltas.csv` | 277 voltas × 81 colunas. Uma linha por volta. |
| `amostras_amostra.csv` | As primeiras 1000 linhas das amostras brutas do UDP, só para mostrar o formato. |
| `metricas_modelos.csv` | MAE, RMSE e R² (CV e teste) de cada modelo, para os três alvos. |
| `importancias.csv` | Importância das features por vários métodos (coeficiente do Lasso, impureza, permutação) e a média. |
| `conclusoes.json` | As conclusões da análise, cada uma com a fonte (arquivo e linhas). |
| `hiperparametros/` | Resultado do tuning: MAE padrão × tunado, melhores parâmetros, importância por permutação, script e logs. |

## Colunas de `voltas.csv`, por grupo

- **Identificação:** `session_uid`, carro, `setup_id` (S0 a S10), número da volta, `fonte_tempo`.
- **Setup (18):** asas, diferencial com e sem aceleração, cambagem, convergência, suspensão, barras estabilizadoras, alturas, pressão e balanço de freio, pressão dos pneus.
- **Contexto:** composto, idade do pneu (`idade_pneu_voltas`), volta do stint, combustível, flashbacks, `volta_saida`, `entrou_box`.
- **Resultado:** `tempo_volta_ms`, tempos de setor, velocidade máxima e média, velocidades mínimas por curva (Sainte-Dévote, Massenet/Cassino, Grampo Loews, Portier, chicane do porto, piscina) e velocidade máxima no túnel.

## Problemas que já conheço

- 4 voltas são de McLaren (só S0); os modelos usam apenas Ferrari e Alfa (273 voltas).
- 42 voltas são de saída e 25 terminam no box, mesmo com `valida=1`.
- 100 voltas têm pelo menos um flashback.
- `desempenho_igual` e `freio_motor` estão 100% vazias, e 12 colunas são constantes.

## Amostras brutas

O arquivo completo de amostras do UDP (`amostras.csv`, cerca de 240 MB) não está no repositório porque passa do limite de 100 MB do GitHub. Ele fica na minha máquina; se precisar dele, me procure e eu disponibilizo por outro meio (Drive, Kaggle ou GitHub Release). Para reproduzir os modelos, `voltas.csv` basta.
