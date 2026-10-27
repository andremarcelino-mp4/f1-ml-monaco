# Notas de revisão: artigo ABNT "Engenheiro de corrida virtual"

**Revisado:** 26 out. 2026  
**Original:** `ARTIGO_ENGENHEIRO_CORRIDA_VIRTUAL.md` (mantido sem alterações)  
**Revisão:** `ARTIGO_ABNT_ENGENHEIRO_CORRIDA_VIRTUAL.md` e `.pdf` (22 páginas, A4)  
**Build:** `build/build_abnt.py` (pandoc + python-docx + LibreOffice). O arquivo `build/ARTIGO_ABNT_ENGENHEIRO_CORRIDA_VIRTUAL.docx` fica disponível como versão editável.

Conferi todos os números contra `voltas.csv`, `conclusoes.json`, `metricas_modelos.csv`, `importancias.csv`, `hiperparametros/*.csv|json` e `api/main.py` (S0/S10). Nenhum dado novo foi inventado, e nenhum DOI foi incluído.

## 1. Erros de conteúdo e consistência numérica

| # | O que estava errado | Correção |
|---|---|---|
| 1 | **Confundimento por combustível incorreto.** O texto dizia que "S4, S7 e S10 rodaram com 5 kg na Alfa Romeo, contra 13 kg no S0" (afirmação herdada de `conclusoes.json`). O cruzamento `combustivel_setup` × carro × acerto no `voltas.csv` mostra outra coisa: na Alfa, **todo** acerto tem 2 ou 3 voltas com 5 kg e as demais entre 12 kg e 14 kg (S4 = 13 kg, S7 = 13 kg e S10 = 14 kg na maioria das voltas). Na Ferrari, o S0 mistura 13 kg e 20 kg, S1 a S4 usam 20 kg e o S7 usa 5 kg. | A Seção 4.4, alínea a, foi reescrita com a distribuição real. |
| 2 | Bases de S0 da Ferrari: o balanço de freio aparecia só como "62 %", mas os dados registram 52, 53 e 62. | Agora consta "52% ou 62%". |
| 3 | O tamanho do *hold-out* aparecia como "≈ 55–57 voltas" em um ponto e como "57 voltas" em outro. Os dois scripts usam a mesma partição (GroupShuffleSplit, seed 42, 273 voltas), que tem 57 voltas. | Padronizado para 57 voltas. |
| 4 | O texto dizia que o MAE de 1,0 s fica "**abaixo** do ruído do piloto". MAE e DP são medidas diferentes, então a comparação direta exagerava o resultado. | Trocado por "da mesma ordem de grandeza"/"próximo ao ruído do piloto" no resumo, na discussão e na conclusão. |
| 5 | Havia ambiguidade setor × acerto: em "+1,50 s, contra −0,03 s no S1 e +0,47 s no S3", S1 e S3 eram **setores**, mas se liam como acertos. O mesmo valia para "S1 −0,41 s; S2 −0,87 s" da Ferrari. | Reescrito como "setor 1/2/3". |
| 6 | O resumo listava ElasticNet e MLP entre os "sete algoritmos" da modelagem principal, mas eles só entram no experimento de *tuning*. | A Seção 2.5 agora separa a modelagem principal do experimento de ajuste. |
| 7 | A correlação "r = 0,86" do setor 2 não dizia o método (é Pearson; o restante usa Spearman). | Indicado "Pearson, r = 0,86". |
| 8 | As 57 sessões do `amostras.csv` e as 40 do `voltas.csv` apareciam lado a lado sem ressalva, e os IDs não batem entre os arquivos. | O texto agora fala em "57 identificadores de sessão" no `amostras.csv`. Nenhuma explicação foi inventada. |
| 9 | "1:14.575" usava ponto decimal, fora do padrão brasileiro. | Substituído por "74,575 s (1min14,575s)". |

Números conferidos e mantidos: as Tabelas 1 a 4 batem integralmente com os CSVs. As médias e os n por acerto foram recalculados no `voltas.csv`. Também foram conferidos o consenso de importâncias (42,7/26,0/15,0/3,4/2,2%), os 11 coeficientes zerados pelo Lasso, os coeficientes, os hiperparâmetros e os ganhos percentuais (−14,9%, −16,5% etc.).

As notas de rodapé sobre divergências entre o plano e a telemetria (S2 30→80, S7 25/45→30/45, S1 50→80), a limpeza descrita na *landing* e não aplicada nos scripts, o ruído de 1,52 s × 1,26 s, o GBM manual × padrão, o Lasso com α = 1 e o "RF 1,03 s" não reproduzível foram **mantidas** e reescritas como notas de rodapé reais.

## 2. Formatação ABNT

| # | Antes | Depois |
|---|---|---|
| 10 | Não havia capa. Faltavam local e natureza do trabalho. | Capa (NBR 14724) com instituição FECAF, curso, autor, título:subtítulo, natureza do trabalho, local e ano 2026. |
| 11 | Não havia sumário. | Sumário (NBR 6027) com pontilhado e páginas calculadas a partir do PDF final. |
| 12 | Não havia **nenhuma citação no texto**, embora as referências estivessem listadas. | Citações no formato autor-data da NBR 10520:2023, por exemplo (Tibshirani, 1996), (Geurts; Ernst; Wehenkel, 2006), (Pedregosa *et al.*, 2011) e (Marcelino, 2026a, 2026b). Foi incluído um parágrafo de contexto teórico na introdução. |
| 13 | As referências não seguiam a NBR 6023 (título em itálico, faixa de páginas com travessão, sem "Acesso em" em parte dos links, URLs entre < >). | NBR 6023:2018: destaque em **negrito**, [*s. l.*], p. 281-305, "Disponível em: … Acesso em: 26 out. 2026.", data provável [2023?] para o SimRacingSetup e 2026a/2026b para as duas obras do autor. |
| 14 | A referência de Spearman estava ausente, embora o teste fosse usado. | Incluído SPEARMAN (1904), artigo clássico real, sem DOI. |
| 15 | As legendas ficavam **abaixo** das figuras, com o padrão "Figura 1 — …" (travessão longo) e sem fonte. | Legenda **acima** ("Figura N – …", meia-risca), imagem centralizada e "Fonte: Elaborada pelo autor (2026), com dados de …" abaixo, em corpo 10. |
| 16 | A tabela textual de acertos era chamada de "Tabela 1". | Passou a ser o **Quadro 1** (dados qualitativos, bordas fechadas). As tabelas foram renumeradas de 1 a 4, no padrão IBGE (sem bordas laterais) e com fonte. |
| 17 | Os pseudotítulos em negrito ("**Q1 —**", "**Ameaças à validade.**", "**Pneus e combustível.**") quebravam a hierarquia. | Foram criadas as seções 4.1 a 4.4. As perguntas Q1 a Q3 e as listas viraram alíneas a), b), c). As seções primárias ficaram em CAIXA ALTA e negrito, e as secundárias em negrito. A estrutura segue o IMRaD (Introdução, Material e Métodos, Resultados, Discussão, Conclusão). |
| 18 | O resumo não tinha abstract, e as palavras-chave incluíam "regressão regularizada". | Resumo com cerca de 250 palavras (NBR 6028), em parágrafo único. Abstract curto em inglês, com keywords. Palavras-chave separadas por ponto e vírgula. |
| 19 | O link do repositório e da *landing* aparecia no cabeçalho do artigo. | Removido do corpo. Os links ficaram só nas referências. |
| 20 | PDF | A4, margens de 3 cm (superior e esquerda) e 2 cm (inferior e direita), Liberation Serif 12 pt, entrelinha de 1,5 no corpo, recuo de 1,25 cm e texto justificado. Resumo, notas, legendas, tabelas e referências usam espaço simples e corpo 10 nas notas, legendas e tabelas. A paginação fica no canto superior direito a partir da 1ª folha textual, e a capa e o sumário não são numerados. As notas são de rodapé reais. |

## 3. Linguagem, ortografia e tipografia

- **Espaço antes de %:** o texto tinha "43 %" e "15,2 %". Agora fica "42,7%", conforme o uso corrente no Brasil. As unidades mantêm o espaço: "1,008 s", "50 km/h", "5 kg".
- **Aproximações e valores:** "≈ 43 %" virou "42,7%", e outros "≈" do resumo e da conclusão foram trocados pelos valores exatos (por exemplo, 3,99 contra 7,10 km/h).
- **Travessões:** o travessão longo foi substituído por meia-risca nas legendas, e o sinal de menos tipográfico (−) foi mantido nos números negativos.
- **Expressões informais ou de marketing removidas:** "clique mágico", "teste 'IA × engenheiro'", "de ponta a ponta", "reta digna do nome", "informação 'faltante'" e "assumiram a liderança". O texto também tinha excesso de negrito de ênfase no corpo, que foi retirado.
- **Notas de figura:** "Gráfico gerado a partir de `resumo_mae.csv`" foi convertido para o formato de fonte ABNT. Não havia trechos do tipo "neste artigo gerado" no original.
- **Ajustes de concordância e regência:** "comparações contra o S0" passou a "comparações com o S0", "Δ vs S0" a "Δ em relação ao S0" e "× " em prosa a "contra".
- **Expoente:** o expoente em Unicode (10⁻¹⁵) ganhou sobrescrito real no PDF.

## 4. Pontos para o autor verificar

- **Local na capa:** coloquei "Taboão da Serra" (sede da FECAF). Ajuste se o campus for outro.
- **URL do repositório:** MARCELINO (2026a) aponta para https://github.com/andremarcelino-mp4/f1-ml-monaco.
- **Data do SimRacingSetup:** a página não informa data, por isso usei [2023?]. Ajuste se encontrar a data.
