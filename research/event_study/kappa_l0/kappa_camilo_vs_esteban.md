# Acuerdo entre anotadores: camilo vs esteban

## Kappa

| Métrica | Clases | n | Kappa | Acuerdo exacto |
|---|---|---|---|---|
| **Primario** (sin `unusable`) | bullish/bearish/neutral | 77 | 0.628 | 75.3% |
| Secundario (con `unusable`) | 4 clases | 100 | 0.608 | 71.0% |

## Tasa de `unusable`

| Anotador | Tasa |
|---|---|
| camilo | 19.0% |
| esteban | 17.0% |

## Matriz de confusión

Filas = `camilo`, columnas = `esteban`.

| | bullish | bearish | neutral | unusable |
|---|---|---|---|---|
| **bullish** | 24 | 1 | 8 | 1 |
| **bearish** | 2 | 13 | 7 | 0 |
| **neutral** | 1 | 0 | 21 | 3 |
| **unusable** | 1 | 1 | 4 | 13 |

## Integridad del cruce

| Control | Valor |
|---|---|
| pares_totales | 100 |
| pares_en_ambos | 100 |
| solo_en_camilo | 0 |
| solo_en_esteban | 0 |
| sin_etiquetar_camilo | 0 |
| sin_etiquetar_esteban | 0 |
| pares_evaluables | 100 |

## Desacuerdos

29 desacuerdos sobre 100 pares evaluables. Ordenados por confianza combinada descendente — los primeros son los que más probablemente señalan un problema del codebook.

| post_id | target_ticker | label_camilo | label_esteban | confianza_camilo | confianza_esteban | confianza_total | clean_text_camilo | nota_camilo | nota_esteban |
|---|---|---|---|---|---|---|---|---|---|
| 372314662 | DDOG | unusable | neutral | 3 | 3 | 6 | long-term forecasting of datadog, inc. ddog 08 25 2021 |  |  |
| 401118485 | DDOG | bullish | neutral | 3 | 3 | 6 | ddog watching for consolidation with ultraalgo , |  |  |
| 50152386 | TGT | neutral | unusable | 3 | 3 | 6 | tgt come network with other investors on -- social wallstreet com |  |  |
| 498629770 | DIS | unusable | bearish | 3 | 3 | 6 | dis strange world another a-z movement bust stay woke go broke definitely not ta |  |  |
| 225021538 | DIS | bullish | neutral | 3 | 3 | 6 | dis ford, disney partner for bronco deal |  |  |
| 257394638 | NCLH | bearish | neutral | 3 | 3 | 6 | hearing norwegian cruise lines offering will be priced between 20.80 and 21.00 r | Norwegian ofrecio un Secondary Public Offering abajo del precio publico de la em |  |
| 405434569 | CRWD | bullish | neutral | 3 | 3 | 6 | just look at that chart! money to be made for both sides. crwd |  |  |
| 52518554 | DIS | bearish | bullish | 3 | 2 | 5 | carlosdcf i did a calendar call spread. but then, that is where we disagree, i s |  |  |
| 498759250 | CRWD | bullish | neutral | 3 | 2 | 5 | crwd guidance . for fy2023, crowdstrike raised the adjusted eps forecast to 1.49 |  |  |
| 482357659 | CRWD | bullish | neutral | 3 | 2 | 5 | measured over the past 5 years, crwd shows a very strong growth in revenue 94.09 |  |  |
| 4604744 | DIS | unusable | neutral | 3 | 2 | 5 | back from a long dis vaca and ended up buying a piece of it. joined the disney v |  |  |
| 45028989 | DIS | bearish | bullish | 3 | 2 | 5 | dis the lucky number based on my calculations, 108.soon .. | El usuario estaba apostando a que disney bajara otros 5 dolares mas |  |
| 407438504 | DIS | bearish | neutral | 3 | 2 | 5 | sweepcast alerted dis with unusual options activity alerted on 135 put expiring  |  |  |
| 102955509 | CMG | bearish | neutral | 3 | 2 | 5 | cmg wait till eod to initiate position. overall this changes nothing till the ac | El CEO de chipotle se acababa de retirar |  |
| 345923507 | NCLH | unusable | neutral | 3 | 2 | 5 | ccl rcl nclh cant wait for my cruise from la to cabo. like a boss! |  |  |
