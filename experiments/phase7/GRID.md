# Fase 7 — GRID (enumerado ANTES de correr)

**Total de tests: 210.** Corrección BH-FDR 5% sobre este total. Etapa 1 solo con datos ≤2021-12-31 (2022 sellado).

Conteo por dimensión: A=42, B=126, C=12, D=10, E=20.

| id | dim | descripción |
|----|-----|-------------|
| A-000 | A | A|TSLA diario: disagreement(t) vs abs_ret(t+1) |
| A-001 | A | A|TSLA diario: disagreement(t) vs range(t+1) |
| A-002 | A | A|TSLA diario: disagreement(t) vs relvol(t+1) |
| A-003 | A | A|TSLA diario: entropy(t) vs abs_ret(t+1) |
| A-004 | A | A|TSLA diario: entropy(t) vs range(t+1) |
| A-005 | A | A|TSLA diario: entropy(t) vs relvol(t+1) |
| A-006 | A | A|AMD diario: disagreement(t) vs abs_ret(t+1) |
| A-007 | A | A|AMD diario: disagreement(t) vs range(t+1) |
| A-008 | A | A|AMD diario: disagreement(t) vs relvol(t+1) |
| A-009 | A | A|AMD diario: entropy(t) vs abs_ret(t+1) |
| A-010 | A | A|AMD diario: entropy(t) vs range(t+1) |
| A-011 | A | A|AMD diario: entropy(t) vs relvol(t+1) |
| A-012 | A | A|DIS diario: disagreement(t) vs abs_ret(t+1) |
| A-013 | A | A|DIS diario: disagreement(t) vs range(t+1) |
| A-014 | A | A|DIS diario: disagreement(t) vs relvol(t+1) |
| A-015 | A | A|DIS diario: entropy(t) vs abs_ret(t+1) |
| A-016 | A | A|DIS diario: entropy(t) vs range(t+1) |
| A-017 | A | A|DIS diario: entropy(t) vs relvol(t+1) |
| A-018 | A | A|BA diario: disagreement(t) vs abs_ret(t+1) |
| A-019 | A | A|BA diario: disagreement(t) vs range(t+1) |
| A-020 | A | A|BA diario: disagreement(t) vs relvol(t+1) |
| A-021 | A | A|BA diario: entropy(t) vs abs_ret(t+1) |
| A-022 | A | A|BA diario: entropy(t) vs range(t+1) |
| A-023 | A | A|BA diario: entropy(t) vs relvol(t+1) |
| A-024 | A | A|GILD diario: disagreement(t) vs abs_ret(t+1) |
| A-025 | A | A|GILD diario: disagreement(t) vs range(t+1) |
| A-026 | A | A|GILD diario: disagreement(t) vs relvol(t+1) |
| A-027 | A | A|GILD diario: entropy(t) vs abs_ret(t+1) |
| A-028 | A | A|GILD diario: entropy(t) vs range(t+1) |
| A-029 | A | A|GILD diario: entropy(t) vs relvol(t+1) |
| A-030 | A | A|TSLA 30min: disagreement(t) vs abs_ret(t+1) |
| A-031 | A | A|TSLA 30min: disagreement(t) vs range(t+1) |
| A-032 | A | A|TSLA 30min: disagreement(t) vs relvol(t+1) |
| A-033 | A | A|TSLA 30min: entropy(t) vs abs_ret(t+1) |
| A-034 | A | A|TSLA 30min: entropy(t) vs range(t+1) |
| A-035 | A | A|TSLA 30min: entropy(t) vs relvol(t+1) |
| A-036 | A | A|AMD 30min: disagreement(t) vs abs_ret(t+1) |
| A-037 | A | A|AMD 30min: disagreement(t) vs range(t+1) |
| A-038 | A | A|AMD 30min: disagreement(t) vs relvol(t+1) |
| A-039 | A | A|AMD 30min: entropy(t) vs abs_ret(t+1) |
| A-040 | A | A|AMD 30min: entropy(t) vs range(t+1) |
| A-041 | A | A|AMD 30min: entropy(t) vs relvol(t+1) |
| B-042 | B | B|TSLA daily: |z_net|>2 -> abs_ret(t+1) |
| B-043 | B | B|TSLA daily: |z_net|>2 -> abs_ret(t+2) |
| B-044 | B | B|TSLA daily: |z_net|>2 -> abs_ret(t+3) |
| B-045 | B | B|TSLA daily: |z_net|>2 -> range(t+1) |
| B-046 | B | B|TSLA daily: |z_net|>2 -> range(t+2) |
| B-047 | B | B|TSLA daily: |z_net|>2 -> range(t+3) |
| B-048 | B | B|TSLA daily: |z_net|>2 -> relvol(t+1) |
| B-049 | B | B|TSLA daily: |z_net|>2 -> relvol(t+2) |
| B-050 | B | B|TSLA daily: |z_net|>2 -> relvol(t+3) |
| B-051 | B | B|TSLA daily: |z_vol|>2 -> abs_ret(t+1) |
| B-052 | B | B|TSLA daily: |z_vol|>2 -> abs_ret(t+2) |
| B-053 | B | B|TSLA daily: |z_vol|>2 -> abs_ret(t+3) |
| B-054 | B | B|TSLA daily: |z_vol|>2 -> range(t+1) |
| B-055 | B | B|TSLA daily: |z_vol|>2 -> range(t+2) |
| B-056 | B | B|TSLA daily: |z_vol|>2 -> range(t+3) |
| B-057 | B | B|TSLA daily: |z_vol|>2 -> relvol(t+1) |
| B-058 | B | B|TSLA daily: |z_vol|>2 -> relvol(t+2) |
| B-059 | B | B|TSLA daily: |z_vol|>2 -> relvol(t+3) |
| B-060 | B | B|AMD daily: |z_net|>2 -> abs_ret(t+1) |
| B-061 | B | B|AMD daily: |z_net|>2 -> abs_ret(t+2) |
| B-062 | B | B|AMD daily: |z_net|>2 -> abs_ret(t+3) |
| B-063 | B | B|AMD daily: |z_net|>2 -> range(t+1) |
| B-064 | B | B|AMD daily: |z_net|>2 -> range(t+2) |
| B-065 | B | B|AMD daily: |z_net|>2 -> range(t+3) |
| B-066 | B | B|AMD daily: |z_net|>2 -> relvol(t+1) |
| B-067 | B | B|AMD daily: |z_net|>2 -> relvol(t+2) |
| B-068 | B | B|AMD daily: |z_net|>2 -> relvol(t+3) |
| B-069 | B | B|AMD daily: |z_vol|>2 -> abs_ret(t+1) |
| B-070 | B | B|AMD daily: |z_vol|>2 -> abs_ret(t+2) |
| B-071 | B | B|AMD daily: |z_vol|>2 -> abs_ret(t+3) |
| B-072 | B | B|AMD daily: |z_vol|>2 -> range(t+1) |
| B-073 | B | B|AMD daily: |z_vol|>2 -> range(t+2) |
| B-074 | B | B|AMD daily: |z_vol|>2 -> range(t+3) |
| B-075 | B | B|AMD daily: |z_vol|>2 -> relvol(t+1) |
| B-076 | B | B|AMD daily: |z_vol|>2 -> relvol(t+2) |
| B-077 | B | B|AMD daily: |z_vol|>2 -> relvol(t+3) |
| B-078 | B | B|DIS daily: |z_net|>2 -> abs_ret(t+1) |
| B-079 | B | B|DIS daily: |z_net|>2 -> abs_ret(t+2) |
| B-080 | B | B|DIS daily: |z_net|>2 -> abs_ret(t+3) |
| B-081 | B | B|DIS daily: |z_net|>2 -> range(t+1) |
| B-082 | B | B|DIS daily: |z_net|>2 -> range(t+2) |
| B-083 | B | B|DIS daily: |z_net|>2 -> range(t+3) |
| B-084 | B | B|DIS daily: |z_net|>2 -> relvol(t+1) |
| B-085 | B | B|DIS daily: |z_net|>2 -> relvol(t+2) |
| B-086 | B | B|DIS daily: |z_net|>2 -> relvol(t+3) |
| B-087 | B | B|DIS daily: |z_vol|>2 -> abs_ret(t+1) |
| B-088 | B | B|DIS daily: |z_vol|>2 -> abs_ret(t+2) |
| B-089 | B | B|DIS daily: |z_vol|>2 -> abs_ret(t+3) |
| B-090 | B | B|DIS daily: |z_vol|>2 -> range(t+1) |
| B-091 | B | B|DIS daily: |z_vol|>2 -> range(t+2) |
| B-092 | B | B|DIS daily: |z_vol|>2 -> range(t+3) |
| B-093 | B | B|DIS daily: |z_vol|>2 -> relvol(t+1) |
| B-094 | B | B|DIS daily: |z_vol|>2 -> relvol(t+2) |
| B-095 | B | B|DIS daily: |z_vol|>2 -> relvol(t+3) |
| B-096 | B | B|BA daily: |z_net|>2 -> abs_ret(t+1) |
| B-097 | B | B|BA daily: |z_net|>2 -> abs_ret(t+2) |
| B-098 | B | B|BA daily: |z_net|>2 -> abs_ret(t+3) |
| B-099 | B | B|BA daily: |z_net|>2 -> range(t+1) |
| B-100 | B | B|BA daily: |z_net|>2 -> range(t+2) |
| B-101 | B | B|BA daily: |z_net|>2 -> range(t+3) |
| B-102 | B | B|BA daily: |z_net|>2 -> relvol(t+1) |
| B-103 | B | B|BA daily: |z_net|>2 -> relvol(t+2) |
| B-104 | B | B|BA daily: |z_net|>2 -> relvol(t+3) |
| B-105 | B | B|BA daily: |z_vol|>2 -> abs_ret(t+1) |
| B-106 | B | B|BA daily: |z_vol|>2 -> abs_ret(t+2) |
| B-107 | B | B|BA daily: |z_vol|>2 -> abs_ret(t+3) |
| B-108 | B | B|BA daily: |z_vol|>2 -> range(t+1) |
| B-109 | B | B|BA daily: |z_vol|>2 -> range(t+2) |
| B-110 | B | B|BA daily: |z_vol|>2 -> range(t+3) |
| B-111 | B | B|BA daily: |z_vol|>2 -> relvol(t+1) |
| B-112 | B | B|BA daily: |z_vol|>2 -> relvol(t+2) |
| B-113 | B | B|BA daily: |z_vol|>2 -> relvol(t+3) |
| B-114 | B | B|GILD daily: |z_net|>2 -> abs_ret(t+1) |
| B-115 | B | B|GILD daily: |z_net|>2 -> abs_ret(t+2) |
| B-116 | B | B|GILD daily: |z_net|>2 -> abs_ret(t+3) |
| B-117 | B | B|GILD daily: |z_net|>2 -> range(t+1) |
| B-118 | B | B|GILD daily: |z_net|>2 -> range(t+2) |
| B-119 | B | B|GILD daily: |z_net|>2 -> range(t+3) |
| B-120 | B | B|GILD daily: |z_net|>2 -> relvol(t+1) |
| B-121 | B | B|GILD daily: |z_net|>2 -> relvol(t+2) |
| B-122 | B | B|GILD daily: |z_net|>2 -> relvol(t+3) |
| B-123 | B | B|GILD daily: |z_vol|>2 -> abs_ret(t+1) |
| B-124 | B | B|GILD daily: |z_vol|>2 -> abs_ret(t+2) |
| B-125 | B | B|GILD daily: |z_vol|>2 -> abs_ret(t+3) |
| B-126 | B | B|GILD daily: |z_vol|>2 -> range(t+1) |
| B-127 | B | B|GILD daily: |z_vol|>2 -> range(t+2) |
| B-128 | B | B|GILD daily: |z_vol|>2 -> range(t+3) |
| B-129 | B | B|GILD daily: |z_vol|>2 -> relvol(t+1) |
| B-130 | B | B|GILD daily: |z_vol|>2 -> relvol(t+2) |
| B-131 | B | B|GILD daily: |z_vol|>2 -> relvol(t+3) |
| B-132 | B | B|TSLA intra: |z_net|>2 -> abs_ret(t+1) |
| B-133 | B | B|TSLA intra: |z_net|>2 -> abs_ret(t+2) |
| B-134 | B | B|TSLA intra: |z_net|>2 -> abs_ret(t+3) |
| B-135 | B | B|TSLA intra: |z_net|>2 -> range(t+1) |
| B-136 | B | B|TSLA intra: |z_net|>2 -> range(t+2) |
| B-137 | B | B|TSLA intra: |z_net|>2 -> range(t+3) |
| B-138 | B | B|TSLA intra: |z_net|>2 -> relvol(t+1) |
| B-139 | B | B|TSLA intra: |z_net|>2 -> relvol(t+2) |
| B-140 | B | B|TSLA intra: |z_net|>2 -> relvol(t+3) |
| B-141 | B | B|TSLA intra: |z_vol|>2 -> abs_ret(t+1) |
| B-142 | B | B|TSLA intra: |z_vol|>2 -> abs_ret(t+2) |
| B-143 | B | B|TSLA intra: |z_vol|>2 -> abs_ret(t+3) |
| B-144 | B | B|TSLA intra: |z_vol|>2 -> range(t+1) |
| B-145 | B | B|TSLA intra: |z_vol|>2 -> range(t+2) |
| B-146 | B | B|TSLA intra: |z_vol|>2 -> range(t+3) |
| B-147 | B | B|TSLA intra: |z_vol|>2 -> relvol(t+1) |
| B-148 | B | B|TSLA intra: |z_vol|>2 -> relvol(t+2) |
| B-149 | B | B|TSLA intra: |z_vol|>2 -> relvol(t+3) |
| B-150 | B | B|AMD intra: |z_net|>2 -> abs_ret(t+1) |
| B-151 | B | B|AMD intra: |z_net|>2 -> abs_ret(t+2) |
| B-152 | B | B|AMD intra: |z_net|>2 -> abs_ret(t+3) |
| B-153 | B | B|AMD intra: |z_net|>2 -> range(t+1) |
| B-154 | B | B|AMD intra: |z_net|>2 -> range(t+2) |
| B-155 | B | B|AMD intra: |z_net|>2 -> range(t+3) |
| B-156 | B | B|AMD intra: |z_net|>2 -> relvol(t+1) |
| B-157 | B | B|AMD intra: |z_net|>2 -> relvol(t+2) |
| B-158 | B | B|AMD intra: |z_net|>2 -> relvol(t+3) |
| B-159 | B | B|AMD intra: |z_vol|>2 -> abs_ret(t+1) |
| B-160 | B | B|AMD intra: |z_vol|>2 -> abs_ret(t+2) |
| B-161 | B | B|AMD intra: |z_vol|>2 -> abs_ret(t+3) |
| B-162 | B | B|AMD intra: |z_vol|>2 -> range(t+1) |
| B-163 | B | B|AMD intra: |z_vol|>2 -> range(t+2) |
| B-164 | B | B|AMD intra: |z_vol|>2 -> range(t+3) |
| B-165 | B | B|AMD intra: |z_vol|>2 -> relvol(t+1) |
| B-166 | B | B|AMD intra: |z_vol|>2 -> relvol(t+2) |
| B-167 | B | B|AMD intra: |z_vol|>2 -> relvol(t+3) |
| C-168 | C | C|TSLA overnight: ov_net vs gap_ret |
| C-169 | C | C|TSLA overnight: ov_net vs gap_abs |
| C-170 | C | C|TSLA overnight: ov_net vs fh_ret |
| C-171 | C | C|TSLA overnight: ov_vol vs gap_ret |
| C-172 | C | C|TSLA overnight: ov_vol vs gap_abs |
| C-173 | C | C|TSLA overnight: ov_vol vs fh_ret |
| C-174 | C | C|AMD overnight: ov_net vs gap_ret |
| C-175 | C | C|AMD overnight: ov_net vs gap_abs |
| C-176 | C | C|AMD overnight: ov_net vs fh_ret |
| C-177 | C | C|AMD overnight: ov_vol vs gap_ret |
| C-178 | C | C|AMD overnight: ov_vol vs gap_abs |
| C-179 | C | C|AMD overnight: ov_vol vs fh_ret |
| D-180 | D | D1|TSLA: corr(net,ret+1) earnings±3d vs normal |
| D-181 | D | D2|TSLA: net pre-earnings -> signo reacción |
| D-182 | D | D1|AMD: corr(net,ret+1) earnings±3d vs normal |
| D-183 | D | D2|AMD: net pre-earnings -> signo reacción |
| D-184 | D | D1|DIS: corr(net,ret+1) earnings±3d vs normal |
| D-185 | D | D2|DIS: net pre-earnings -> signo reacción |
| D-186 | D | D1|BA: corr(net,ret+1) earnings±3d vs normal |
| D-187 | D | D2|BA: net pre-earnings -> signo reacción |
| D-188 | D | D1|GILD: corr(net,ret+1) earnings±3d vs normal |
| D-189 | D | D2|GILD: net pre-earnings -> signo reacción |
| E-190 | E | E|net(TSLA) vs ret+1(AMD) |
| E-191 | E | E|net(TSLA) vs ret+1(DIS) |
| E-192 | E | E|net(TSLA) vs ret+1(BA) |
| E-193 | E | E|net(TSLA) vs ret+1(GILD) |
| E-194 | E | E|net(AMD) vs ret+1(TSLA) |
| E-195 | E | E|net(AMD) vs ret+1(DIS) |
| E-196 | E | E|net(AMD) vs ret+1(BA) |
| E-197 | E | E|net(AMD) vs ret+1(GILD) |
| E-198 | E | E|net(DIS) vs ret+1(TSLA) |
| E-199 | E | E|net(DIS) vs ret+1(AMD) |
| E-200 | E | E|net(DIS) vs ret+1(BA) |
| E-201 | E | E|net(DIS) vs ret+1(GILD) |
| E-202 | E | E|net(BA) vs ret+1(TSLA) |
| E-203 | E | E|net(BA) vs ret+1(AMD) |
| E-204 | E | E|net(BA) vs ret+1(DIS) |
| E-205 | E | E|net(BA) vs ret+1(GILD) |
| E-206 | E | E|net(GILD) vs ret+1(TSLA) |
| E-207 | E | E|net(GILD) vs ret+1(AMD) |
| E-208 | E | E|net(GILD) vs ret+1(DIS) |
| E-209 | E | E|net(GILD) vs ret+1(BA) |
