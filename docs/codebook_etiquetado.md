# Codebook de etiquetado manual — corpus del event study

Guía de anotación para los lotes de `data/manual_labels/`. Anotadores:
**camilo** y **esteban**.

Este documento fija los criterios ANTES de etiquetar. Si durante el trabajo
aparece un caso que no está acá, no lo resuelvan por su cuenta: anótenlo en la
columna `nota`, sigan adelante, y se decide entre los dos al cerrar el lote
`l0`. Un criterio cambiado a mitad de camino invalida el kappa.

---

## 1. Qué se etiqueta: el par (post, ticker)

La unidad NO es el mensaje. Es el par **(post, ticker objetivo)**.

Un mensaje que menciona dos tickers del estudio aparece **dos veces**, en dos
filas distintas, con distinto `target_ticker`. Cada fila se juzga por separado:

| post_id | target_ticker | clean_text | label |
|---|---|---|---|
| 90412 | DIS | selling dis to get into cmg | `bearish` |
| 90412 | CMG | selling dis to get into cmg | `bullish` |

> **La regla más importante de todo el codebook:** etiquetás el sentimiento
> **hacia `target_ticker`**, no el ánimo general del mensaje. Es el error más
> común y el que más daño le hace al kappa.

Mirá siempre la columna `target_ticker` antes de leer el texto.

### Los 6 tickers del estudio

`NCLH` (cruceros) · `DIS` (Disney) · `CRWD` (CrowdStrike) · `TGT` (Target) ·
`CMG` (Chipotle) · `DDOG` (Datadog)

---

## 2. El texto que ves está limpiado — leelo sabiendo qué se perdió

`clean_text` pasó por el pipeline de `src/sentiment/preprocessor.py`. Esto **no**
es el mensaje original. Lo que se fue:

| Se elimina | Consecuencia al etiquetar |
|---|---|
| **Emojis** (🚀 💀 🌙) | Un post que era puro emoji llega vacío de señal |
| **El `$` de los cashtags** | `$DIS` → `dis`, indistinguible de una palabra común |
| **Mayúsculas** | Todo en minúscula; se pierde el énfasis de GRITAR |
| **URLs y markdown** | Desaparece el link a la noticia que daba contexto |
| **Boilerplate** (`not financial advice`) | Puede dejar comas o frases colgando |

Ejemplos reales del pipeline (verificados corriéndolo):

```
"$DIS breaking out! Parks reopening is the catalyst 🚀🚀"
  → "dis breaking out! parks reopening is the catalyst"

"$CRWD 🚀🚀🚀"
  → "crwd"                          ← toda la señal se perdió

"$DDOG $NET $SNOW"
  → "ddog net snow"                 ← parece una frase, son 3 tickers

"$CMG down 3% but I'm holding, not financial advice"
  → "cmg down 3% but i'm holding,"  ← coma colgando del boilerplate
```

**Lo que SÍ sobrevive y sirve:** signos `! ? % . , ' ( ) -`, números,
porcentajes, y toda la jerga (`moon`, `bagholder`, `lmao`, `puts`, `calls`).

Dos trampas que salen de acá:

- **`dis` puede ser Disney o el slang "dis"** (faltar el respeto). Resolvé por
  contexto; si no se puede, `unusable`.
- **`net`, `snow`, `tgt`** se leen como palabras comunes. Si la frase solo tiene
  tickers pegados, es una lista de símbolos, no una oración.

---

## 3. Las cuatro etiquetas

Se escriben en minúscula, exactamente así:

| Valor | Cuándo |
|---|---|
| `bullish` | El autor expresa postura **alcista** sobre `target_ticker` |
| `bearish` | El autor expresa postura **bajista** sobre `target_ticker` |
| `neutral` | Hay contenido sobre el ticker, pero **sin postura direccional** |
| `unusable` | **No se puede juzgar** el sentimiento hacia el ticker |

### Árbol de decisión

Aplicalo en orden, de arriba hacia abajo:

1. ¿El texto tiene contenido más allá de tickers sueltos?
   **No** → `unusable`
2. ¿El texto dice algo sobre `target_ticker` en particular?
   **No** (habla solo de otro ticker / del mercado en general) → `unusable`
3. ¿El autor toma postura direccional sobre `target_ticker`?
   **No** (dato, noticia, pregunta sin inclinación) → `neutral`
4. ¿La postura es de subida o de bajada?
   → `bullish` / `bearish`

### `neutral` vs `unusable`: la distinción que más se confunde

- `neutral` = **se pudo leer, y no hay postura.**
  *"tgt earnings tomorrow"* → informativo, sin dirección.
- `unusable` = **no se pudo leer la postura, punto.**
  *"crwd"* → no hay nada que juzgar.

`unusable` **no es "no sé"**. Si dudás entre `bullish` y `bearish`, elegí una y
bajá `confianza` a 1. Marcar `unusable` por duda te saca del kappa primario y
distorsiona la métrica.

---

## 4. Columnas a llenar

Se tocan **solo estas cuatro**. Las cinco primeras (`post_id`,
`target_ticker`, `tickers_detectados`, `fecha`, `clean_text`) **no se editan
nunca** — son la llave del cruce entre anotadores.

### `label` (obligatoria)
`bullish` | `bearish` | `neutral` | `unusable`

### `confianza` (obligatoria) — escala 1 a 3

| | Significado |
|---|---|
| **3** | Sin duda. Cualquiera lo leería igual. |
| **2** | Bastante seguro, pero admite otra lectura. |
| **1** | Dudo de verdad; elegí la más probable. |

`kappa_calculator.py` ordena los desacuerdos por confianza combinada: los
casos donde **ambos pusieron 3 y aun así difieren** son los que revelan fallas
del codebook. Por eso la escala tiene que significar lo mismo para los dos.

### `base` (obligatoria) — en qué te apoyaste

| Valor | Cuándo |
|---|---|
| `texto` | Lo dice explícitamente (*"breaking out"*, *"will never recover"*) |
| `jerga` | Depende de slang de foro (*"to the moon"*, *"bagholder"*, *"done"*) |
| `cifra` | La señal es un número o resultado (*"down 3%"*, *"beat earnings"*) |
| `contexto` | Requiere saber algo externo (*"parks reopening"* es bueno para DIS) |

Para `unusable` se escribe `—`.

> No hay valor `emoji`: el pipeline los elimina antes de que los veas.

### `nota` (opcional)
Texto libre. Usala para casos raros, para justificar un `unusable`, o para
marcar algo a discutir. Es la columna que hace productiva la reunión de cierre
de `l0`.

---

## 5. Casos borde ya resueltos

Estos están decididos. No los re-litiguen fila por fila.

**1. El ticker objetivo no aparece en el texto.**
Pasa: en los mensajes con label nativo, el ticker sale de los metadatos, no
siempre de un cashtag en el cuerpo. Si el texto no permite saber que habla de
ese ticker → `unusable`.

**2. Pregunta sin inclinación.**
*"is dis a buy here? thinking about it"* → `neutral`.
Pero *"is dis a buy here? looks like free money"* → `bullish`: la segunda parte
sí toma postura.

**3. Sarcasmo e ironía.**
Etiquetá la **intención**, no la literalidad. *"nclh lmao this is done"* es
`bearish` con `base=jerga`, aunque "done" en sí no sea negativo.

**4. Postura del autor, no la tuya.**
Si alguien es alcista sobre NCLH en marzo 2020 y vos sabés cómo terminó, es
igual `bullish`. No etiquetás si acertó; etiquetás qué dijo.

**5. Hecho con signo pero sin postura.**
*"crwd beat earnings, raised guidance. still not buying at this multiple"* →
`bearish`: el dato es bueno, pero el autor cierra con postura negativa. Manda
la postura del autor.
En cambio *"crwd beat earnings, raised guidance"* solo → `neutral` (dato sin
postura). Si el resultado es claramente bueno/malo y el autor lo presenta como
tal sin más, `cifra` + la dirección del hecho.

**6. Mención de paso o comparación.**
*"ddog net snow"* con `target_ticker=DDOG` → `unusable`: es una lista.
*"prefiero ddog antes que snow"* con `target_ticker=DDOG` → `bullish`;
con `target_ticker=SNOW` no aplica (SNOW no está en el estudio).

**7. Spam, promoción, otro idioma, texto truncado.**
→ `unusable`, `base=—`, y una `nota` breve.

**8. Postura sobre el mercado, no sobre el ticker.**
*"dis everything is crashing today"* → `unusable` si habla del mercado entero
y no de DIS en particular.

**9. Posiciones y opciones.**
*"bought more ddog today"* → `bullish` (`base=texto`).
*"puts on tgt"* → `bearish` (`base=jerga`).
*"holding cmg"* sin más → `neutral`: mantener no es tomar postura nueva.

**10. Precio objetivo sin dirección.**
*"tgt 250"* → `unusable` si no se sabe si lo ve como techo o piso.

---

## 6. Ejemplo completo de cómo se llena

Así se ve un tramo ya etiquetado:

| post_id | target_ticker | fecha | clean_text | label | confianza | base | nota |
|---|---|---|---|---|---|---|---|
| 90412 | DIS | 2021-03-04 | selling dis to get into cmg | bearish | 3 | texto | |
| 90412 | CMG | 2021-03-04 | selling dis to get into cmg | bullish | 3 | texto | mismo post, otro ticker |
| 71233 | NCLH | 2020-08-19 | nclh lmao this is done | bearish | 3 | jerga | "done" = se acabó |
| 55019 | TGT | 2021-11-02 | tgt earnings tomorrow | neutral | 3 | texto | informativo |
| 33871 | CRWD | 2022-01-14 | crwd | unusable | 3 | — | sin contenido |
| 44120 | DDOG | 2022-05-03 | ddog net snow | unusable | 3 | — | lista de tickers |
| 61887 | DIS | 2021-03-04 | dis breaking out! parks reopening is the catalyst | bullish | 3 | contexto | |
| 28345 | CMG | 2022-06-10 | cmg down 3% but i'm holding, | neutral | 2 | cifra | baja el precio, no la postura |
| 19022 | TGT | 2021-05-18 | tgt to the moon!!! | bullish | 3 | jerga | |
| 77104 | NCLH | 2020-04-02 | nclh cruise lines will never recover... bagholders beware | bearish | 3 | texto | |

Fijate en `28345`: el precio bajó (`cifra`), pero el autor dice que mantiene.
No declara postura nueva → `neutral` con `confianza=2`, porque admite lectura
alcista ("holding" como convicción).

---

## 7. Reparto y orden de trabajo

Son **1000 pares** en 4 lotes → **650 para cada uno**:

| Lote | Pares | ¿Mismos pares para ambos? | Para qué |
|---|---|---|---|
| `l0` | 100 | **Sí** | Calibración |
| `doble` | 200 | **Sí** | Kappa primario (va a la tesis) |
| `test` | 300 | No — 150 c/u | Evaluación |
| `train` | 400 | No — 200 c/u | Entrenar la cabeza lineal |

Archivos: `data/manual_labels/{lote}_{camilo|esteban}.csv`.

En los lotes compartidos cada uno recibe **los mismos pares en orden distinto**
(semilla derivada por anotador, para evitar sesgo de orden). No se asusten si
los archivos no coinciden fila a fila: el cruce es por
`(post_id, target_ticker)`, no por posición.

### Orden obligatorio

1. **`l0` primero, los dos, sin hablarlo.** 100 pares cada uno.
2. **Corran el kappa y siéntense a revisar.** Miren los desacuerdos ordenados
   por confianza y las `nota`. Actualicen este codebook con lo que aparezca.
3. **Recién ahí:** `doble`, `test`, `train`.

`l0` existe para descubrir en qué difieren **antes** de que ese desacuerdo
contamine los 900 pares restantes. Si lo saltean, el kappa de `doble` va a
medir confusión de criterios en vez de dificultad real.

### Medir el acuerdo

```powershell
python -m research.event_study.kappa_calculator `
    data/manual_labels/l0_camilo.csv `
    data/manual_labels/l0_esteban.csv `
    experiments/kappa
```

Reporta kappa primario (3 clases, sin `unusable`), kappa secundario (4 clases),
matriz de confusión, tasa de `unusable` por anotador, y los desacuerdos
ordenados por confianza combinada.

Referencia de lectura (Landis & Koch): <0.40 pobre · 0.40–0.60 moderado ·
0.60–0.80 sustancial · >0.80 casi perfecto. Si `l0` da por debajo de 0.60, el
problema es el codebook, no los anotadores: arréglenlo y repitan antes de
seguir.

---

## 8. Qué NO hacer

- **No** etiquetar el ánimo del post en vez del sentimiento hacia
  `target_ticker`.
- **No** editar `post_id`, `target_ticker`, `tickers_detectados`, `fecha` ni
  `clean_text`. Se rompe el cruce.
- **No** reordenar, borrar ni agregar filas.
- **No** consultarse durante `l0`. Contamina la medición.
- **No** usar `unusable` como "no sé". Es "no se puede juzgar".
- **No** buscar el mensaje original ni qué hizo el precio después. Se etiqueta
  con lo que está en `clean_text`.
- **No** dejar que Excel reformatee la columna `fecha` al guardar
  (`2021-03-04` → `04/03/2021`). Los `post_id` son de 9 dígitos y sobreviven
  bien, pero si usan Excel importen todas las columnas como **texto** y
  guarden como CSV UTF-8. Google Sheets y LibreOffice dan menos problemas.

---

## 9. Registro de cambios

Toda modificación posterior a `l0` se anota acá, con fecha y motivo.

| Fecha | Cambio | Motivo |
|---|---|---|
| 2026-09-11 | Versión inicial | Pre-registro, antes de `l0` |
