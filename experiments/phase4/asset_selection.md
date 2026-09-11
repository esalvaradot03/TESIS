# Fase 4 — Tarea 1: selección data-driven de activos

Rango: **2015-01-01 … 2022-12-31**. Universo cruzado StockTwits∩FNSPID: **7,276 tickers**. Se muestran top 30 por rank combinado + ETFs de referencia.

Marcas: `trunc_2020` = el precio local FNSPID termina en 2019/2020 (artefacto de truncamiento); `yf_ok` = yfinance cubre 2015→2022 completo.

| # | ticker | tipo | sector | mktcap $M | ST msgs | ST/día | news | news/día | combo | fnspid_end | yf_ok | trunc2020 |
|---|--------|------|--------|-----------|---------|--------|------|----------|-------|-----------|-------|-----------|
| 1 | **BABA** | ACCION | Consumer Cyclica | 282,088 | 483,678 | 166.1 | 10,385 | 4.6 | 17 | 2023-12-28 | sí |  |
| 2 | **AMD** | ACCION | Technology | 862,816 | 1,184,974 | 413.0 | 6,979 | 4.1 | 18 | 2023-12-28 | sí |  |
| 3 | **QQQ** | ETF | Large Growth | — | 660,480 | 226.1 | 7,130 | 3.4 | 22 | 2023-12-28 | sí |  |
| 4 | **SPY** | ETF | Large Blend | — | 4,539,879 | 1,553.7 | 5,928 | 5.9 | 27 | 2023-12-28 | sí |  |
| 5 | **GME** | ACCION | Consumer Cyclica | 9,988 | 1,007,966 | 412.3 | 6,071 | 3.6 | 31 | 2023-12-28 | sí |  |
| 6 | **TSLA** | ACCION | Consumer Cyclica | 1,481,483 | 2,483,170 | 850.1 | 5,044 | 9.6 | 44 | 2023-12-28 | sí |  |
| 7 | **MU** | ACCION | Technology | 1,021,288 | 258,931 | 90.7 | 7,265 | 3.9 | 49 | 2023-12-28 | sí |  |
| 8 | **NVDA** | ACCION | Technology | 5,146,963 | 395,391 | 143.3 | 5,554 | 4.1 | 51 | 2023-12-28 | sí |  |
| 9 | **AMC** | ACCION | Communication Se | 1,848 | 5,817,215 | 2,728.5 | 4,511 | 3.0 | 55 | 2023-12-28 | sí |  |
| 10 | **BA** | ACCION | Industrials | 171,945 | 324,153 | 116.6 | 5,658 | 5.2 | 57 | 2023-12-28 | sí |  |
| 11 | **DIS** | ACCION | Communication Se | 168,702 | 214,366 | 74.0 | 7,574 | 6.1 | 60 | 2023-12-28 | sí |  |
| 12 | **NIO** | ACCION | Consumer Cyclica | 12,604 | 962,299 | 606.7 | 4,083 | 3.7 | 81 | 2023-12-28 | no |  |
| 13 | **GE** | ACCION | Industrials | 376,504 | 147,406 | 52.7 | 6,747 | 3.2 | 85 | 2023-12-28 | sí |  |
| 14 | **NVAX** | ACCION | Healthcare | 1,386 | 299,471 | 106.2 | 4,320 | 3.0 | 92 | 2023-12-28 | sí |  |
| 15 | **SNAP** | ACCION | Communication Se | 7,888 | 314,280 | 138.9 | 4,188 | 3.0 | 94 | — | no | ⚠ |
| 16 | **AAPL** | ACCION | Technology | 4,810,109 | 1,270,996 | 435.0 | 3,567 | 12.3 | 101 | 2023-12-28 | sí |  |
| 17 | **AAL** | ACCION | Industrials | 10,337 | 130,115 | 48.4 | 5,902 | 3.2 | 107 | 2023-12-28 | sí |  |
| 18 | **GILD** | ACCION | Healthcare | 163,515 | 101,133 | 35.6 | 9,638 | 4.2 | 119 | 2023-12-28 | sí |  |
| 19 | **INTC** | ACCION | Technology | 517,628 | 101,551 | 35.8 | 9,043 | 5.7 | 120 | 2023-12-28 | sí |  |
| 20 | **PYPL** | ACCION | Financial Servic | 48,974 | 115,633 | 43.9 | 5,039 | 2.8 | 139 | 2023-12-28 | sí |  |
| 21 | **GOOG** | ACCION | Communication Se | 4,517,514 | 85,867 | 29.6 | 7,619 | 5.5 | 144 | 2023-12-28 | sí |  |
| 22 | **WMT** | ACCION | Consumer Defensi | 895,523 | 88,656 | 31.4 | 6,859 | 5.0 | 145 | 2023-12-28 | sí |  |
| 23 | **NFLX** | ACCION | Communication Se | 310,252 | 370,661 | 127.0 | 3,028 | 3.3 | 162 | 2020-06-19 | sí | ⚠ |
| 24 | **T** | ACCION | Communication Se | 148,903 | 69,414 | 25.3 | 8,278 | 4.1 | 167 | 2023-12-28 | sí |  |
| 25 | **GPRO** | ACCION | Technology | 122 | 108,302 | 39.6 | 3,813 | 3.1 | 199 | 2023-12-28 | sí |  |
| 26 | **CGC** | ACCION | Healthcare | 427 | 129,689 | 76.9 | 3,173 | 2.7 | 202 | 2023-12-28 | sí |  |
| 27 | **XOM** | ACCION | Energy | 598,986 | 60,825 | 21.7 | 5,368 | 4.5 | 208 | 2023-12-28 | sí |  |
| 28 | **INO** | ACCION | Healthcare | 96 | 354,900 | 127.1 | 2,726 | 2.8 | 213 | 2023-12-28 | sí |  |
| 29 | **GLD** | ETF | Commodities Focu | — | 87,731 | 30.1 | 3,864 | 2.1 | 219 | 2023-12-28 | sí |  |
| 30 | **JD** | ACCION | Consumer Cyclica | 39,543 | 108,986 | 40.6 | 3,204 | 2.7 | 222 | 2023-12-28 | sí |  |
| 31 | **USO** | ETF | Commodities Focu | — | 80,345 | 27.9 | 3,071 | 1.7 | 280 | 2023-12-28 | sí |  |
| 32 | **IWM** | ETF | Small Blend | — | 102,879 | 35.4 | 798 | 1.6 | 1,335 | 2020-06-19 | sí | ⚠ |
