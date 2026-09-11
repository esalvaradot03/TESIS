# Fase 7 — Resultados: búsqueda exploratoria estructurada (dos etapas)

> Diseño anti p-hacking: Etapa 1 exploratoria solo con datos ≤2021-12-31 (2022 sellado), BH-FDR 5% sobre toda la grilla; Etapa 2 confirmatoria una sola vez sobre 2022.

## Conteo honesto

- **210 tests** corridos en la grilla (ver GRID.md).
- **84 pasaron FDR 5%** en train (umbral p ≤ 0.01947).
- **23 confirmaron** en el test sellado 2022.

## Etapa 1 — top 15 por p crudo (de la grilla completa)

| id | dim | descripción | n | stat | p crudo | p BH | pasa FDR |
|----|-----|-------------|---|------|---------|------|----------|
| C-172 | C | C|TSLA overnight: ov_vol vs gap_abs | 362 | +0.5638 | 9.2e-32 | 1.9e-29 | sí |
| A-037 | A | A|AMD 30min: disagreement(t) vs range(t+1) | 4,326 | +0.1583 | 1.2e-25 | 1.2e-23 | sí |
| B-093 | B | B|DIS daily: |z_vol|>2 -> relvol(t+1) | 124 | +0.5711 | 9.9e-25 | 7.0e-23 | sí |
| A-031 | A | A|TSLA 30min: disagreement(t) vs range(t+1) | 4,327 | +0.1496 | 4.6e-23 | 2.0e-21 | sí |
| B-147 | B | B|TSLA intra: |z_vol|>2 -> relvol(t+1) | 244 | +0.4253 | 4.8e-23 | 2.0e-21 | sí |
| C-168 | C | C|TSLA overnight: ov_net vs gap_ret | 362 | +0.4797 | 3.2e-22 | 1.1e-20 | sí |
| B-075 | B | B|AMD daily: |z_vol|>2 -> relvol(t+1) | 102 | +0.5932 | 6.9e-21 | 2.1e-19 | sí |
| A-040 | A | A|AMD 30min: entropy(t) vs range(t+1) | 4,326 | +0.1398 | 2.5e-20 | 6.5e-19 | sí |
| A-034 | A | A|TSLA 30min: entropy(t) vs range(t+1) | 4,327 | +0.1355 | 3.6e-19 | 8.4e-18 | sí |
| B-057 | B | B|TSLA daily: |z_vol|>2 -> relvol(t+1) | 100 | +0.5269 | 8.5e-15 | 1.8e-13 | sí |
| B-076 | B | B|AMD daily: |z_vol|>2 -> relvol(t+2) | 102 | +0.3574 | 3.2e-13 | 6.0e-12 | sí |
| B-129 | B | B|GILD daily: |z_vol|>2 -> relvol(t+1) | 89 | +0.2708 | 7.4e-12 | 1.3e-10 | sí |
| A-022 | A | A|BA diario: entropy(t) vs range(t+1) | 1,762 | +0.1572 | 3.2e-11 | 5.2e-10 | sí |
| B-094 | B | B|DIS daily: |z_vol|>2 -> relvol(t+2) | 124 | +0.2012 | 4.1e-11 | 6.2e-10 | sí |
| A-016 | A | A|DIS diario: entropy(t) vs range(t+1) | 1,762 | +0.1518 | 1.5e-10 | 2.0e-09 | sí |

## Etapa 1 — sobrevivientes FDR

| id | descripción | n | stat | p crudo | p BH |
|----|-------------|---|------|---------|------|
| C-172 | C|TSLA overnight: ov_vol vs gap_abs | 362 | +0.5638 | 9.2e-32 | 1.9e-29 |
| A-037 | A|AMD 30min: disagreement(t) vs range(t+1) | 4,326 | +0.1583 | 1.2e-25 | 1.2e-23 |
| B-093 | B|DIS daily: |z_vol|>2 -> relvol(t+1) | 124 | +0.5711 | 9.9e-25 | 7.0e-23 |
| A-031 | A|TSLA 30min: disagreement(t) vs range(t+1) | 4,327 | +0.1496 | 4.6e-23 | 2.0e-21 |
| B-147 | B|TSLA intra: |z_vol|>2 -> relvol(t+1) | 244 | +0.4253 | 4.8e-23 | 2.0e-21 |
| C-168 | C|TSLA overnight: ov_net vs gap_ret | 362 | +0.4797 | 3.2e-22 | 1.1e-20 |
| B-075 | B|AMD daily: |z_vol|>2 -> relvol(t+1) | 102 | +0.5932 | 6.9e-21 | 2.1e-19 |
| A-040 | A|AMD 30min: entropy(t) vs range(t+1) | 4,326 | +0.1398 | 2.5e-20 | 6.5e-19 |
| A-034 | A|TSLA 30min: entropy(t) vs range(t+1) | 4,327 | +0.1355 | 3.6e-19 | 8.4e-18 |
| B-057 | B|TSLA daily: |z_vol|>2 -> relvol(t+1) | 100 | +0.5269 | 8.5e-15 | 1.8e-13 |
| B-076 | B|AMD daily: |z_vol|>2 -> relvol(t+2) | 102 | +0.3574 | 3.2e-13 | 6.0e-12 |
| B-129 | B|GILD daily: |z_vol|>2 -> relvol(t+1) | 89 | +0.2708 | 7.4e-12 | 1.3e-10 |
| A-022 | A|BA diario: entropy(t) vs range(t+1) | 1,762 | +0.1572 | 3.2e-11 | 5.2e-10 |
| B-094 | B|DIS daily: |z_vol|>2 -> relvol(t+2) | 124 | +0.2012 | 4.1e-11 | 6.2e-10 |
| A-016 | A|DIS diario: entropy(t) vs range(t+1) | 1,762 | +0.1518 | 1.5e-10 | 2.0e-09 |
| A-013 | A|DIS diario: disagreement(t) vs range(t+1) | 1,762 | +0.1518 | 1.5e-10 | 2.0e-09 |
| B-165 | B|AMD intra: |z_vol|>2 -> relvol(t+1) | 256 | +0.3185 | 4.5e-10 | 5.5e-09 |
| B-090 | B|DIS daily: |z_vol|>2 -> range(t+1) | 124 | +0.0055 | 5.5e-10 | 6.5e-09 |
| A-030 | A|TSLA 30min: disagreement(t) vs abs_ret(t+1) | 4,327 | +0.0928 | 9.5e-10 | 1.0e-08 |
| A-019 | A|BA diario: disagreement(t) vs range(t+1) | 1,762 | +0.1411 | 2.7e-09 | 2.8e-08 |
| B-166 | B|AMD intra: |z_vol|>2 -> relvol(t+2) | 245 | +0.1832 | 9.5e-09 | 9.2e-08 |
| A-036 | A|AMD 30min: disagreement(t) vs abs_ret(t+1) | 4,326 | +0.0871 | 9.6e-09 | 9.2e-08 |
| B-144 | B|TSLA intra: |z_vol|>2 -> range(t+1) | 244 | +0.0019 | 1.2e-08 | 1.1e-07 |
| B-072 | B|AMD daily: |z_vol|>2 -> range(t+1) | 102 | +0.0163 | 5.2e-08 | 4.6e-07 |
| B-148 | B|TSLA intra: |z_vol|>2 -> relvol(t+2) | 225 | +0.2104 | 6.0e-08 | 5.0e-07 |
| B-087 | B|DIS daily: |z_vol|>2 -> abs_ret(t+1) | 124 | +0.0052 | 6.3e-08 | 5.1e-07 |
| C-178 | C|AMD overnight: ov_vol vs gap_abs | 362 | +0.2757 | 9.8e-08 | 7.4e-07 |
| A-039 | A|AMD 30min: entropy(t) vs abs_ret(t+1) | 4,326 | +0.0809 | 9.9e-08 | 7.4e-07 |
| C-174 | C|AMD overnight: ov_net vs gap_ret | 362 | +0.2669 | 2.5e-07 | 1.8e-06 |
| B-111 | B|BA daily: |z_vol|>2 -> relvol(t+1) | 90 | +0.2178 | 3.8e-07 | 2.6e-06 |
| B-073 | B|AMD daily: |z_vol|>2 -> range(t+2) | 102 | +0.0136 | 5.5e-07 | 3.7e-06 |
| B-167 | B|AMD intra: |z_vol|>2 -> relvol(t+3) | 231 | +0.1547 | 1.3e-06 | 8.4e-06 |
| B-058 | B|TSLA daily: |z_vol|>2 -> relvol(t+2) | 100 | +0.2273 | 5.5e-06 | 3.5e-05 |
| A-010 | A|AMD diario: entropy(t) vs range(t+1) | 1,762 | +0.1050 | 1.0e-05 | 6.2e-05 |
| A-033 | A|TSLA 30min: entropy(t) vs abs_ret(t+1) | 4,327 | +0.0661 | 1.3e-05 | 8.0e-05 |
| A-007 | A|AMD diario: disagreement(t) vs range(t+1) | 1,762 | +0.1026 | 1.6e-05 | 9.3e-05 |
| A-041 | A|AMD 30min: entropy(t) vs relvol(t+1) | 4,318 | +0.0614 | 5.3e-05 | 3.0e-04 |
| B-054 | B|TSLA daily: |z_vol|>2 -> range(t+1) | 100 | +0.0059 | 5.5e-05 | 3.0e-04 |
| B-126 | B|GILD daily: |z_vol|>2 -> range(t+1) | 89 | +0.0034 | 7.9e-05 | 4.0e-04 |
| B-123 | B|GILD daily: |z_vol|>2 -> abs_ret(t+1) | 89 | +0.0044 | 7.9e-05 | 4.0e-04 |
| B-077 | B|AMD daily: |z_vol|>2 -> relvol(t+3) | 102 | +0.1730 | 8.0e-05 | 4.0e-04 |
| B-162 | B|AMD intra: |z_vol|>2 -> range(t+1) | 256 | +0.0011 | 8.0e-05 | 4.0e-04 |
| B-138 | B|TSLA intra: |z_net|>2 -> relvol(t+1) | 184 | +0.1472 | 9.1e-05 | 4.5e-04 |
| B-069 | B|AMD daily: |z_vol|>2 -> abs_ret(t+1) | 102 | +0.0090 | 1.5e-04 | 7.4e-04 |
| B-146 | B|TSLA intra: |z_vol|>2 -> range(t+3) | 208 | +0.0014 | 2.2e-04 | 9.9e-04 |
| B-091 | B|DIS daily: |z_vol|>2 -> range(t+2) | 124 | +0.0027 | 2.2e-04 | 9.9e-04 |
| B-051 | B|TSLA daily: |z_vol|>2 -> abs_ret(t+1) | 100 | +0.0078 | 2.3e-04 | 0.001 |
| B-163 | B|AMD intra: |z_vol|>2 -> range(t+2) | 245 | +0.0011 | 2.4e-04 | 0.001 |
| B-145 | B|TSLA intra: |z_vol|>2 -> range(t+2) | 225 | +0.0014 | 3.0e-04 | 0.001 |
| B-063 | B|AMD daily: |z_net|>2 -> range(t+1) | 97 | +0.0068 | 3.1e-04 | 0.001 |
| A-021 | A|BA diario: entropy(t) vs abs_ret(t+1) | 1,762 | +0.0857 | 3.2e-04 | 0.001 |
| A-001 | A|TSLA diario: disagreement(t) vs range(t+1) | 1,762 | +0.0847 | 3.7e-04 | 0.001 |
| A-004 | A|TSLA diario: entropy(t) vs range(t+1) | 1,762 | +0.0847 | 3.7e-04 | 0.001 |
| A-018 | A|BA diario: disagreement(t) vs abs_ret(t+1) | 1,762 | +0.0823 | 5.5e-04 | 0.002 |
| B-164 | B|AMD intra: |z_vol|>2 -> range(t+3) | 231 | +0.0006 | 7.4e-04 | 0.003 |
| B-059 | B|TSLA daily: |z_vol|>2 -> relvol(t+3) | 100 | +0.1301 | 8.8e-04 | 0.003 |
| B-150 | B|AMD intra: |z_net|>2 -> abs_ret(t+1) | 245 | +0.0008 | 9.5e-04 | 0.004 |
| B-112 | B|BA daily: |z_vol|>2 -> relvol(t+2) | 90 | +0.1395 | 0.001 | 0.004 |
| B-070 | B|AMD daily: |z_vol|>2 -> abs_ret(t+2) | 102 | +0.0064 | 0.001 | 0.004 |
| A-012 | A|DIS diario: disagreement(t) vs abs_ret(t+1) | 1,762 | +0.0760 | 0.001 | 0.005 |
| A-015 | A|DIS diario: entropy(t) vs abs_ret(t+1) | 1,762 | +0.0760 | 0.001 | 0.005 |
| B-130 | B|GILD daily: |z_vol|>2 -> relvol(t+2) | 89 | +0.1174 | 0.002 | 0.006 |
| B-084 | B|DIS daily: |z_net|>2 -> relvol(t+1) | 80 | +0.1704 | 0.002 | 0.006 |
| B-082 | B|DIS daily: |z_net|>2 -> range(t+2) | 80 | +0.0027 | 0.003 | 0.009 |
| B-135 | B|TSLA intra: |z_net|>2 -> range(t+1) | 184 | +0.0017 | 0.003 | 0.009 |
| B-064 | B|AMD daily: |z_net|>2 -> range(t+2) | 97 | +0.0095 | 0.003 | 0.010 |
| C-175 | C|AMD overnight: ov_net vs gap_abs | 362 | -0.1504 | 0.004 | 0.013 |
| B-066 | B|AMD daily: |z_net|>2 -> relvol(t+1) | 97 | +0.1289 | 0.004 | 0.013 |
| B-149 | B|TSLA intra: |z_vol|>2 -> relvol(t+3) | 208 | +0.1282 | 0.005 | 0.014 |
| C-171 | C|TSLA overnight: ov_vol vs gap_ret | 362 | +0.1472 | 0.005 | 0.015 |
| B-095 | B|DIS daily: |z_vol|>2 -> relvol(t+3) | 124 | +0.0857 | 0.006 | 0.017 |
| B-081 | B|DIS daily: |z_net|>2 -> range(t+1) | 80 | +0.0031 | 0.006 | 0.019 |
| B-156 | B|AMD intra: |z_net|>2 -> relvol(t+1) | 245 | +0.0799 | 0.007 | 0.019 |
| B-153 | B|AMD intra: |z_net|>2 -> range(t+1) | 245 | +0.0006 | 0.007 | 0.019 |
| B-113 | B|BA daily: |z_vol|>2 -> relvol(t+3) | 90 | +0.1256 | 0.007 | 0.021 |
| B-141 | B|TSLA intra: |z_vol|>2 -> abs_ret(t+1) | 244 | +0.0005 | 0.009 | 0.024 |
| B-048 | B|TSLA daily: |z_net|>2 -> relvol(t+1) | 80 | +0.1269 | 0.010 | 0.028 |
| B-074 | B|AMD daily: |z_vol|>2 -> range(t+3) | 102 | +0.0069 | 0.011 | 0.029 |
| B-159 | B|AMD intra: |z_vol|>2 -> abs_ret(t+1) | 256 | +0.0005 | 0.012 | 0.033 |
| A-000 | A|TSLA diario: disagreement(t) vs abs_ret(t+1) | 1,762 | +0.0579 | 0.015 | 0.039 |
| A-003 | A|TSLA diario: entropy(t) vs abs_ret(t+1) | 1,762 | +0.0579 | 0.015 | 0.039 |
| B-065 | B|AMD daily: |z_net|>2 -> range(t+3) | 97 | +0.0063 | 0.017 | 0.042 |
| A-009 | A|AMD diario: entropy(t) vs abs_ret(t+1) | 1,762 | +0.0558 | 0.019 | 0.049 |
| C-176 | C|AMD overnight: ov_net vs fh_ret | 363 | -0.1226 | 0.019 | 0.049 |

## Etapa 2 — confirmación sobre 2022 (sellado)

| id | descripción | dir train | stat test | p test | placebo p | veredicto |
|----|-------------|-----------|-----------|--------|-----------|-----------|
| C-172 | C|TSLA overnight: ov_vol vs gap_abs | + | +0.3414 | 2.9e-08 | 0.002 | **CONFIRMA** |
| A-037 | A|AMD 30min: disagreement(t) vs range(t+1) | + | -0.0064 | 0.724 | 0.721 | no |
| B-093 | B|DIS daily: |z_vol|>2 -> relvol(t+1) | + | +0.3622 | 2.9e-04 | 0.002 | **CONFIRMA** |
| A-031 | A|TSLA 30min: disagreement(t) vs range(t+1) | + | +0.0910 | 5.8e-07 | 0.002 | **CONFIRMA** |
| B-147 | B|TSLA intra: |z_vol|>2 -> relvol(t+1) | + | +0.3966 | 1.2e-09 | 0.002 | **CONFIRMA** |
| C-168 | C|TSLA overnight: ov_net vs gap_ret | + | +0.3482 | 1.5e-08 | 0.002 | **CONFIRMA** |
| B-075 | B|AMD daily: |z_vol|>2 -> relvol(t+1) | + | +0.3091 | 0.011 | 0.002 | no |
| A-040 | A|AMD 30min: entropy(t) vs range(t+1) | + | -0.0022 | 0.905 | 0.908 | no |
| A-034 | A|TSLA 30min: entropy(t) vs range(t+1) | + | +0.0910 | 5.8e-07 | 0.002 | **CONFIRMA** |
| B-057 | B|TSLA daily: |z_vol|>2 -> relvol(t+1) | + | +0.4615 | 0.004 | 0.002 | **CONFIRMA** |
| B-076 | B|AMD daily: |z_vol|>2 -> relvol(t+2) | + | +0.0466 | 0.521 | 0.589 | no |
| B-129 | B|GILD daily: |z_vol|>2 -> relvol(t+1) | + | +0.1045 | 0.042 | 0.248 | no |
| A-022 | A|BA diario: entropy(t) vs range(t+1) | + | +0.2087 | 9.0e-04 | 0.002 | **CONFIRMA** |
| B-094 | B|DIS daily: |z_vol|>2 -> relvol(t+2) | + | +0.1521 | 0.004 | 0.080 | no |
| A-016 | A|DIS diario: entropy(t) vs range(t+1) | + | +0.1421 | 0.025 | 0.020 | no |
| A-013 | A|DIS diario: disagreement(t) vs range(t+1) | + | +0.1421 | 0.025 | 0.020 | no |
| B-165 | B|AMD intra: |z_vol|>2 -> relvol(t+1) | + | +0.1541 | 7.6e-05 | 0.002 | **CONFIRMA** |
| B-090 | B|DIS daily: |z_vol|>2 -> range(t+1) | + | +0.0056 | 0.118 | 0.030 | no |
| A-030 | A|TSLA 30min: disagreement(t) vs abs_ret(t+1) | + | +0.0485 | 0.008 | 0.008 | **CONFIRMA** |
| A-019 | A|BA diario: disagreement(t) vs range(t+1) | + | +0.2087 | 9.0e-04 | 0.002 | **CONFIRMA** |
| B-166 | B|AMD intra: |z_vol|>2 -> relvol(t+2) | + | +0.0947 | 7.1e-04 | 0.006 | **CONFIRMA** |
| A-036 | A|AMD 30min: disagreement(t) vs abs_ret(t+1) | + | +0.0009 | 0.963 | 0.966 | no |
| B-144 | B|TSLA intra: |z_vol|>2 -> range(t+1) | + | +0.0023 | 1.8e-05 | 0.002 | **CONFIRMA** |
| B-072 | B|AMD daily: |z_vol|>2 -> range(t+1) | + | +0.0082 | 0.151 | 0.104 | no |
| B-148 | B|TSLA intra: |z_vol|>2 -> relvol(t+2) | + | +0.1997 | 1.1e-05 | 0.002 | **CONFIRMA** |
| B-087 | B|DIS daily: |z_vol|>2 -> abs_ret(t+1) | + | +0.0082 | 0.210 | 0.084 | no |
| C-178 | C|AMD overnight: ov_vol vs gap_abs | + | +0.2106 | 7.8e-04 | 0.004 | **CONFIRMA** |
| A-039 | A|AMD 30min: entropy(t) vs abs_ret(t+1) | + | +0.0024 | 0.896 | 0.888 | no |
| C-174 | C|AMD overnight: ov_net vs gap_ret | + | +0.3728 | 1.1e-09 | 0.002 | **CONFIRMA** |
| B-111 | B|BA daily: |z_vol|>2 -> relvol(t+1) | + | +0.3219 | 0.036 | 0.006 | no |
| B-073 | B|AMD daily: |z_vol|>2 -> range(t+2) | + | -0.0052 | 0.504 | 0.305 | no |
| B-167 | B|AMD intra: |z_vol|>2 -> relvol(t+3) | + | +0.1054 | 0.003 | 0.008 | **CONFIRMA** |
| B-058 | B|TSLA daily: |z_vol|>2 -> relvol(t+2) | + | +0.0627 | 0.497 | 0.439 | no |
| A-010 | A|AMD diario: entropy(t) vs range(t+1) | + | -0.1501 | 0.018 | 0.018 | no |
| A-033 | A|TSLA 30min: entropy(t) vs abs_ret(t+1) | + | +0.0485 | 0.008 | 0.008 | **CONFIRMA** |
| A-007 | A|AMD diario: disagreement(t) vs range(t+1) | + | -0.1501 | 0.018 | 0.018 | no |
| A-041 | A|AMD 30min: entropy(t) vs relvol(t+1) | + | +0.0182 | 0.318 | 0.341 | no |
| B-054 | B|TSLA daily: |z_vol|>2 -> range(t+1) | + | +0.0119 | 0.279 | 0.084 | no |
| B-126 | B|GILD daily: |z_vol|>2 -> range(t+1) | + | +0.0020 | 0.095 | 0.313 | no |
| B-123 | B|GILD daily: |z_vol|>2 -> abs_ret(t+1) | + | +0.0000 | 0.442 | 1.000 | no |
| B-077 | B|AMD daily: |z_vol|>2 -> relvol(t+3) | + | +0.0206 | 0.737 | 0.808 | no |
| B-162 | B|AMD intra: |z_vol|>2 -> range(t+1) | + | +0.0019 | 4.0e-05 | 0.002 | **CONFIRMA** |
| B-138 | B|TSLA intra: |z_net|>2 -> relvol(t+1) | + | +0.0532 | 0.191 | 0.251 | no |
| B-069 | B|AMD daily: |z_vol|>2 -> abs_ret(t+1) | + | -0.0054 | 0.699 | 0.457 | no |
| B-146 | B|TSLA intra: |z_vol|>2 -> range(t+3) | + | +0.0004 | 0.627 | 0.313 | no |
| B-091 | B|DIS daily: |z_vol|>2 -> range(t+2) | + | +0.0030 | 0.227 | 0.212 | no |
| B-051 | B|TSLA daily: |z_vol|>2 -> abs_ret(t+1) | + | +0.0069 | 0.678 | 0.473 | no |
| B-163 | B|AMD intra: |z_vol|>2 -> range(t+2) | + | +0.0011 | 9.2e-04 | 0.018 | no |
| B-145 | B|TSLA intra: |z_vol|>2 -> range(t+2) | + | +0.0008 | 0.004 | 0.056 | no |
| B-063 | B|AMD daily: |z_net|>2 -> range(t+1) | + | +0.0005 | 0.976 | 0.946 | no |
| A-021 | A|BA diario: entropy(t) vs abs_ret(t+1) | + | +0.0825 | 0.193 | 0.194 | no |
| A-001 | A|TSLA diario: disagreement(t) vs range(t+1) | + | +0.1964 | 0.002 | 0.002 | **CONFIRMA** |
| A-004 | A|TSLA diario: entropy(t) vs range(t+1) | + | +0.1964 | 0.002 | 0.002 | **CONFIRMA** |
| A-018 | A|BA diario: disagreement(t) vs abs_ret(t+1) | + | +0.0825 | 0.193 | 0.194 | no |
| B-164 | B|AMD intra: |z_vol|>2 -> range(t+3) | + | +0.0014 | 0.003 | 0.004 | **CONFIRMA** |
| B-059 | B|TSLA daily: |z_vol|>2 -> relvol(t+3) | + | +0.0026 | 0.972 | 0.970 | no |
| B-150 | B|AMD intra: |z_net|>2 -> abs_ret(t+1) | + | -0.0001 | 0.641 | 0.798 | no |
| B-112 | B|BA daily: |z_vol|>2 -> relvol(t+2) | + | +0.0893 | 0.245 | 0.445 | no |
| B-070 | B|AMD daily: |z_vol|>2 -> abs_ret(t+2) | + | -0.0060 | 0.294 | 0.363 | no |
| A-012 | A|DIS diario: disagreement(t) vs abs_ret(t+1) | + | +0.1122 | 0.077 | 0.086 | no |
| A-015 | A|DIS diario: entropy(t) vs abs_ret(t+1) | + | +0.1121 | 0.077 | 0.090 | no |
| B-130 | B|GILD daily: |z_vol|>2 -> relvol(t+2) | + | +0.0085 | 0.276 | 0.976 | no |
| B-084 | B|DIS daily: |z_net|>2 -> relvol(t+1) | + | -0.0509 | 0.608 | 0.555 | no |
| B-082 | B|DIS daily: |z_net|>2 -> range(t+2) | + | +0.0007 | 0.853 | 0.774 | no |
| B-135 | B|TSLA intra: |z_net|>2 -> range(t+1) | + | -0.0003 | 0.906 | 0.579 | no |
| B-064 | B|AMD daily: |z_net|>2 -> range(t+2) | + | +0.0018 | 0.707 | 0.733 | no |
| C-175 | C|AMD overnight: ov_net vs gap_abs | − | -0.0662 | 0.296 | 0.289 | no |
| B-066 | B|AMD daily: |z_net|>2 -> relvol(t+1) | + | +0.0114 | 0.843 | 0.896 | no |
| B-149 | B|TSLA intra: |z_vol|>2 -> relvol(t+3) | + | +0.0937 | 0.134 | 0.022 | no |
| C-171 | C|TSLA overnight: ov_vol vs gap_ret | + | -0.0522 | 0.410 | 0.407 | no |
| B-095 | B|DIS daily: |z_vol|>2 -> relvol(t+3) | + | +0.2253 | 0.020 | 0.028 | no |
| B-081 | B|DIS daily: |z_net|>2 -> range(t+1) | + | +0.0024 | 0.626 | 0.351 | no |
| B-156 | B|AMD intra: |z_net|>2 -> relvol(t+1) | + | +0.0101 | 0.846 | 0.792 | no |
| B-153 | B|AMD intra: |z_net|>2 -> range(t+1) | + | +0.0001 | 0.888 | 0.882 | no |
| B-113 | B|BA daily: |z_vol|>2 -> relvol(t+3) | + | +0.1230 | 0.115 | 0.311 | no |
| B-141 | B|TSLA intra: |z_vol|>2 -> abs_ret(t+1) | + | +0.0022 | 4.3e-04 | 0.002 | **CONFIRMA** |
| B-048 | B|TSLA daily: |z_net|>2 -> relvol(t+1) | + | +0.0028 | 0.139 | 1.000 | no |
| B-074 | B|AMD daily: |z_vol|>2 -> range(t+3) | + | -0.0055 | 0.179 | 0.285 | no |
| B-159 | B|AMD intra: |z_vol|>2 -> abs_ret(t+1) | + | +0.0005 | 0.140 | 0.299 | no |
| A-000 | A|TSLA diario: disagreement(t) vs abs_ret(t+1) | + | +0.0884 | 0.163 | 0.174 | no |
| A-003 | A|TSLA diario: entropy(t) vs abs_ret(t+1) | + | +0.0884 | 0.163 | 0.174 | no |
| B-065 | B|AMD daily: |z_net|>2 -> range(t+3) | + | -0.0042 | 0.325 | 0.499 | no |
| A-009 | A|AMD diario: entropy(t) vs abs_ret(t+1) | + | -0.1461 | 0.021 | 0.036 | no |
| C-176 | C|AMD overnight: ov_net vs fh_ret | − | -0.0536 | 0.398 | 0.375 | no |

## Incremental sobre baseline autorregresivo (dimensión A, requerido)

Para los pares A que pasaron FDR: ¿la dispersión agrega AUC sobre un baseline autorregresivo del propio target (ret/range/volumen son persistentes)? ΔAUC en train (split interno 70/30).

| id | descripción | AUC baseline AR | AUC AR+dispersión | ΔAUC |
|----|-------------|-----------------|-------------------|------|
| A-037 | A|AMD 30min: disagreement(t) vs range(t+1) | 0.8629 | 0.8622 | -0.0007 |
| A-031 | A|TSLA 30min: disagreement(t) vs range(t+1) | 0.8579 | 0.8599 | +0.0020 |
| A-040 | A|AMD 30min: entropy(t) vs range(t+1) | 0.8629 | 0.8623 | -0.0006 |
| A-034 | A|TSLA 30min: entropy(t) vs range(t+1) | 0.8579 | 0.8600 | +0.0021 |
| A-022 | A|BA diario: entropy(t) vs range(t+1) | 0.7872 | 0.7868 | -0.0004 |
| A-016 | A|DIS diario: entropy(t) vs range(t+1) | 0.7933 | 0.8037 | +0.0104 |
| A-013 | A|DIS diario: disagreement(t) vs range(t+1) | 0.7933 | 0.8037 | +0.0104 |
| A-030 | A|TSLA 30min: disagreement(t) vs abs_ret(t+1) | 0.6753 | 0.6734 | -0.0019 |
| A-019 | A|BA diario: disagreement(t) vs range(t+1) | 0.7872 | 0.7898 | +0.0025 |
| A-036 | A|AMD 30min: disagreement(t) vs abs_ret(t+1) | 0.6526 | 0.6557 | +0.0031 |
| A-039 | A|AMD 30min: entropy(t) vs abs_ret(t+1) | 0.6526 | 0.6560 | +0.0034 |
| A-010 | A|AMD diario: entropy(t) vs range(t+1) | 0.7137 | 0.7111 | -0.0026 |
| A-033 | A|TSLA 30min: entropy(t) vs abs_ret(t+1) | 0.6753 | 0.6705 | -0.0048 |
| A-007 | A|AMD diario: disagreement(t) vs range(t+1) | 0.7137 | 0.7152 | +0.0015 |
| A-041 | A|AMD 30min: entropy(t) vs relvol(t+1) | 0.8007 | 0.7975 | -0.0032 |
| A-021 | A|BA diario: entropy(t) vs abs_ret(t+1) | 0.6139 | 0.6120 | -0.0018 |
| A-001 | A|TSLA diario: disagreement(t) vs range(t+1) | 0.7684 | 0.7728 | +0.0044 |
| A-004 | A|TSLA diario: entropy(t) vs range(t+1) | 0.7684 | 0.7728 | +0.0044 |
| A-018 | A|BA diario: disagreement(t) vs abs_ret(t+1) | 0.6139 | 0.6141 | +0.0003 |
| A-012 | A|DIS diario: disagreement(t) vs abs_ret(t+1) | 0.5836 | 0.5896 | +0.0060 |
| A-015 | A|DIS diario: entropy(t) vs abs_ret(t+1) | 0.5836 | 0.5896 | +0.0060 |
| A-000 | A|TSLA diario: disagreement(t) vs abs_ret(t+1) | 0.5500 | 0.5569 | +0.0069 |
| A-003 | A|TSLA diario: entropy(t) vs abs_ret(t+1) | 0.5500 | 0.5569 | +0.0069 |
| A-009 | A|AMD diario: entropy(t) vs abs_ret(t+1) | 0.5571 | 0.5535 | -0.0036 |

**ΔAUC medio = +0.0021.** Si es ~0, la dispersión no agrega sobre la persistencia del propio target (la correlación marginal es el acoplamiento contemporáneo dispersión↔volatilidad montado sobre el clustering).

## Lectura

**23/84 sobrevivientes confirmaron** en el test sellado 2022 (de 210 tests totales).
- **21 confirmaciones son de ACTIVIDAD** (target = range / relvol / abs_ret = volatilidad o volumen), NO dirección de retorno. Son esperables: volatilidad y volumen son fuertemente persistentes y el sentimiento (dispersión, shocks de volumen de mensajes) se acopla a ellos. La tabla incremental-A muestra que sobre el baseline autorregresivo el aporte es marginal.
- **2 confirmaciones tocan DIRECCIÓN/gap:** principalmente `overnight ov_net → gap_ret` (TSLA C-168 ρ_test≈0.35, AMD C-174 ρ_test≈0.37). El sentimiento **fuera de horario predice el signo/magnitud del gap de apertura** — dimensión nunca probada antes (en Fase 6 se excluía el overnight). Es el hallazgo más interesante, con la salvedad de que el sentimiento overnight es probablemente **coincidente** con las noticias que causan el gap (ambos reaccionan a lo mismo), más que anticiparlo.
- **Ningún efecto confirma predicción de la dirección del retorno intradía o diario** más allá del gap de apertura — consistente con Fases 1-6.
