# Acuerdo entre anotadores: camilo vs esteban

## Kappa

| Métrica | Clases | n | Kappa | Acuerdo exacto |
|---|---|---|---|---|
| **Primario** (sin `unusable`) | bullish/bearish/neutral | 155 | 0.788 | 86.5% |
| Secundario (con `unusable`) | 4 clases | 200 | 0.739 | 81.0% |

## Tasa de `unusable`

| Anotador | Tasa |
|---|---|
| camilo | 18.5% |
| esteban | 18.0% |

## Matriz de confusión

Filas = `camilo`, columnas = `esteban`.

| | bullish | bearish | neutral | unusable |
|---|---|---|---|---|
| **bullish** | 67 | 2 | 2 | 1 |
| **bearish** | 2 | 38 | 3 | 2 |
| **neutral** | 6 | 6 | 29 | 5 |
| **unusable** | 5 | 3 | 1 | 28 |

## Integridad del cruce

| Control | Valor |
|---|---|
| pares_totales | 200 |
| pares_en_ambos | 200 |
| solo_en_camilo | 0 |
| solo_en_esteban | 0 |
| sin_etiquetar_camilo | 0 |
| sin_etiquetar_esteban | 0 |
| pares_evaluables | 200 |

## Desacuerdos

38 desacuerdos sobre 200 pares evaluables. Ordenados por confianza combinada descendente — los primeros son los que más probablemente señalan un problema del codebook.

| post_id | target_ticker | label_camilo | label_esteban | confianza_camilo | confianza_esteban | confianza_total | clean_text_camilo | nota_camilo | nota_esteban |
|---|---|---|---|---|---|---|---|---|---|
| 35946499 | DIS | neutral | unusable | 3 | 3 | 6 | 10 45am stocks buzz dis jd jrjc lh |  |  |
| 458678897 | DIS | unusable | bullish | 3 | 3 | 6 | dis wow |  |  |
| 2656343 | CMG | bullish | unusable | 3 | 3 | 6 | cmg finally moving out of the shadows today. |  |  |
| 25568610 | TGT | neutral | unusable | 3 | 3 | 6 | august 8 2014 chart patterns yhoo gfi tgt tibx clf iag bby swx two |  |  |
| 39982060 | CMG | neutral | unusable | 3 | 3 | 6 | end of day scan over bollinger band aapl abc cprx mrge ery amri cmg mnst fgen al | informativo scan |  |
| 352464740 | CRWD | neutral | bearish | 3 | 3 | 6 | (video analysis - 20 trade ideas for the week ahead) tech absolutely ripped high | lista video |  |
| 455303665 | DDOG | neutral | bearish | 3 | 3 | 6 | insider madre armelle de reports selling 5,625 shares of ddog for a total cost o | informativo |  |
| 324061719 | DDOG | neutral | bullish | 3 | 3 | 6 | ddog news for datadog? 6.18%? |  | no es predicción, ya pasó |
| 494985205 | DIS | unusable | bullish | 3 | 3 | 6 | ftt-x dis nrbo hudi mack ' yet, another successful day in the stock market. 32,7 | spam |  |
| 73463545 | CMG | bearish | neutral | 3 | 3 | 6 | cmg reported 4q16 net income of 15.98mm or 0.55 per share, down from 67.87mm or  |  |  |
| 138072563 | CMG | bearish | neutral | 3 | 3 | 6 | cmg wage pressure builds in restaurant sector sep. 20, 2018 9 39 am et by clark  |  |  |
| 487661677 | TGT | neutral | bearish | 3 | 2 | 5 | here's what has retail stocks wmt, tgt, and cost slipping today! | informativo |  |
| 423643669 | TGT | neutral | unusable | 3 | 2 | 5 | wmt price action not news is what is propelling this higher - resistance tapers  | habla de WMT principalmente |  |
| 394018196 | CMG | neutral | bearish | 2 | 3 | 5 | cmg so many bears |  |  |
| 3911001 | CMG | neutral | bearish | 3 | 2 | 5 | cmg filed yesterday for a 3.6 m stock offering, didn't hear a word about it anyw |  |  |
