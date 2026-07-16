# Etapa A.2 — Viabilidad point-in-time

## 1. Limpieza del universo (candidatos con >=200 días)

- survivors (precio reciente): **665** | deslistados: **345** | sin precio: **880**
- Clasificación quoteType de survivors: {'EQUITY': 475, 'UNKNOWN': 132, 'ETF': 58}
- **Equities reales (survivors): 475** | cripto en deslistados: 0 | deslistados presunto-equity: 345

## 2. Market cap point-in-time

- FNSPID full_history trae shares outstanding: **NO** (solo OHLCV).
- Fuente local con shares/fundamentals en D:\trading-data: **ninguna**.
- yfinance para deslistados: **sin datos** (posibly delisted).
- **=> Market cap point-in-time NO es reconstruible. Mitigación B (por cap) NO implementable.**

## 3. Proxy de liquidez point-in-time (dollar volume mediano, FNSPID)

- Survivors equity (n=475): p10 $87K | p50 $2.3M | p90 $100.9M
- Deslistados equity (n=344): p10 $67K | p50 $2.0M | p90 $81.2M

## 4. Retorno de delisting

- Los **345** deslistados tienen precio hasta su último día (FNSPID).
- Distribución del año del último precio: {2020: 345}
- Razón del delisting (quiebra vs fusión vs corte de datos): **NO identificable** con los datos disponibles.
