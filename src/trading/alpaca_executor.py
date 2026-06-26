"""
Ejecutor de órdenes en Alpaca Paper Trading.

Recibe señales del modelo XGBoost y gestiona el portfolio:
  1. Obtiene el estado actual de la cuenta y posiciones abiertas.
  2. Cierra posiciones cuya señal pasó a 0 (o no aparecen en las señales).
  3. Abre posiciones nuevas para señales = 1, respetando el límite de
     MAX_OPEN_POSITIONS posiciones simultáneas.
  4. Si hay más de MAX_OPEN_POSITIONS señales activas, selecciona las de
     mayor confianza (probabilidad del modelo).
  5. Dimensiona cada posición de forma igual-ponderada:
       notional = equity_disponible / n_posiciones_objetivo
  6. Registra cada operación individualmente con timestamp, símbolo,
     dirección, monto y estado de la orden.

Solo ejecuta órdenes de mercado ('market', time_in_force='day').
Validación de mercado abierto antes de cada ejecución.
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import NamedTuple

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from config.settings import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_FEED,
    MAX_OPEN_POSITIONS,
    MIN_ORDER_NOTIONAL,
)

logger = logging.getLogger(__name__)

# Pausa entre órdenes individuales (segundos) para evitar ráfagas al API
_ORDER_SLEEP: float = 0.25
# Reintentos por orden en caso de error transitorio (429, 5xx)
_MAX_ORDER_RETRIES: int = 3
_RETRY_BACKOFF: float = 2.0


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

class SignalRecord(NamedTuple):
    """Señal generada por el modelo para un ticker en un día concreto."""
    ticker: str
    signal: int          # 1 = comprar / mantener, 0 = cerrar / no operar
    confidence: float    # probabilidad de la clase "sube" (para ranking)


@dataclass
class OrderRecord:
    """Registro de una orden enviada a Alpaca."""
    timestamp: str
    ticker: str
    action: str          # "BUY" | "SELL" | "CLOSE"
    notional: float      # importe en USD (0 para cierres por position)
    qty: float | None    # cantidad de acciones (None si se usó notional)
    alpaca_order_id: str
    status: str          # "submitted" | "failed" | "skipped"
    error: str = ""


@dataclass
class ExecutionSummary:
    """Resumen completo de una ejecución de señales."""
    timestamp: str
    portfolio_equity: float
    buying_power: float
    positions_before: int
    positions_after: int       # estimado
    signals_received: int
    buy_signals_selected: int  # tras aplicar el límite de MAX_OPEN_POSITIONS
    positions_opened: int
    positions_closed: int
    positions_maintained: int
    orders_failed: int
    target_notional_per_position: float
    orders: list[OrderRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serializa a dict para logging/JSON."""
        d = asdict(self)
        d["orders"] = [asdict(o) for o in self.orders]
        return d


# ---------------------------------------------------------------------------
# Cliente Alpaca
# ---------------------------------------------------------------------------

def _build_trading_client() -> TradingClient:
    """Construye el cliente de trading (paper=True) de alpaca-py."""
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise EnvironmentError(
            "ALPACA_API_KEY y ALPACA_API_SECRET deben estar definidos en .env"
        )
    return TradingClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_API_SECRET,
        paper=True,
    )


def _build_data_client() -> StockHistoricalDataClient:
    """Construye el cliente de datos históricos de alpaca-py."""
    return StockHistoricalDataClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_API_SECRET,
    )


# ---------------------------------------------------------------------------
# Normalización de símbolos
# ---------------------------------------------------------------------------

def _to_alpaca(ticker: str) -> str:
    """Convierte formato interno ('BRK-B') al símbolo Alpaca ('BRK/B')."""
    return ticker.replace("-", "/")


def _from_alpaca(symbol: str) -> str:
    """Convierte símbolo Alpaca ('BRK/B') al formato interno ('BRK-B')."""
    return symbol.replace("/", "-")


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class AlpacaExecutor:
    """
    Gestiona el portfolio de paper trading en Alpaca a partir de señales del modelo.

    El cliente REST se crea una sola vez y se reutiliza.
    """

    def __init__(
        self,
        max_positions: int = MAX_OPEN_POSITIONS,
        min_order_notional: float = MIN_ORDER_NOTIONAL,
    ) -> None:
        """
        Args:
            max_positions: Máximo de posiciones largas simultáneas.
            min_order_notional: Importe mínimo por orden en USD.
        """
        self._client = _build_trading_client()
        self._data_client = _build_data_client()
        self._max_positions = max_positions
        self._min_notional = min_order_notional
        logger.info(
            "AlpacaExecutor inicializado. max_positions=%d | min_notional=$%.2f | paper=True",
            max_positions, min_order_notional,
        )

    # ------------------------------------------------------------------
    # Consultas al estado del bróker
    # ------------------------------------------------------------------

    def _get_account(self) -> dict:
        """Devuelve equity y buying_power de la cuenta como floats."""
        acc = self._client.get_account()
        return {
            "equity": float(acc.equity),
            "buying_power": float(acc.buying_power),
            "cash": float(acc.cash),
            "portfolio_value": float(acc.portfolio_value),
        }

    def _get_open_positions(self) -> dict[str, float]:
        """
        Devuelve {ticker_interno: qty_float} de todas las posiciones abiertas.

        Convierte símbolos Alpaca al formato interno (BRK/B → BRK-B).
        """
        positions = self._client.get_all_positions()
        return {
            _from_alpaca(p.symbol): float(p.qty)
            for p in positions
        }

    def _is_market_open(self) -> bool:
        """Devuelve True si el mercado NYSE está abierto ahora mismo."""
        clock = self._client.get_clock()
        return bool(clock.is_open)

    # ------------------------------------------------------------------
    # Selección de señales (aplica límite de posiciones)
    # ------------------------------------------------------------------

    def _select_buy_signals(
        self,
        signals: list[SignalRecord],
    ) -> list[SignalRecord]:
        """
        Filtra señales de compra y aplica el límite MAX_OPEN_POSITIONS.

        Cuando hay más señales de compra que posiciones disponibles, retiene
        las de mayor confianza (probabilidad del modelo). En caso de empate,
        ordena alfabéticamente para reproducibilidad.

        Args:
            signals: Lista completa de señales del modelo para el día.

        Returns:
            Lista de señales de compra seleccionadas (longitud ≤ max_positions).
        """
        buy = [s for s in signals if s.signal == 1]
        buy_sorted = sorted(buy, key=lambda s: (-s.confidence, s.ticker))
        selected = buy_sorted[: self._max_positions]

        if len(buy) > self._max_positions:
            logger.info(
                "Señales de compra (%d) > MAX_OPEN_POSITIONS (%d). "
                "Seleccionadas las %d de mayor confianza.",
                len(buy), self._max_positions, self._max_positions,
            )
        return selected

    # ------------------------------------------------------------------
    # Ejecución de órdenes individuales
    # ------------------------------------------------------------------

    def _submit_order(
        self,
        ticker: str,
        side: str,           # "buy" | "sell"
        notional: float | None = None,
        qty: float | None = None,
    ) -> OrderRecord:
        """
        Envía una orden de mercado a Alpaca con reintentos ante errores transitorios.

        Se usa 'notional' (importe en USD) para position sizing proporcional.
        Si Alpaca rechaza la orden notional (acción no fraccional), reintenta
        calculando la cantidad entera más cercana desde el último precio.

        Args:
            ticker: Símbolo en formato interno.
            side: "buy" o "sell".
            notional: Importe en USD (para fractional / notional orders).
            qty: Cantidad de acciones (fallback si notional falla).

        Returns:
            OrderRecord con el resultado de la operación.
        """
        alpaca_symbol = _to_alpaca(ticker)
        ts = datetime.now(tz=timezone.utc).isoformat()
        action = side.upper()
        backoff = _RETRY_BACKOFF

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        for attempt in range(1, _MAX_ORDER_RETRIES + 1):
            try:
                if notional is not None:
                    order_data = MarketOrderRequest(
                        symbol=alpaca_symbol,
                        notional=round(notional, 2),
                        side=order_side,
                        time_in_force=TimeInForce.DAY,
                    )
                else:
                    order_data = MarketOrderRequest(
                        symbol=alpaca_symbol,
                        qty=qty,
                        side=order_side,
                        time_in_force=TimeInForce.DAY,
                    )

                order = self._client.submit_order(order_data)
                record = OrderRecord(
                    timestamp=ts,
                    ticker=ticker,
                    action=action,
                    notional=notional or 0.0,
                    qty=qty,
                    alpaca_order_id=order.id,
                    status="submitted",
                )
                logger.info(
                    "ORDEN %-5s | %-10s | $%10.2f | id=%s",
                    action, ticker, notional or 0.0, order.id,
                )
                return record

            except APIError as exc:
                status_code = getattr(exc, "status_code", None)

                # Error 422: acción no fraccional → reintentar con qty entera
                if status_code == 422 and notional is not None and attempt == 1:
                    logger.warning(
                        "%s no admite órdenes notional. Calculando qty entera...", ticker
                    )
                    try:
                        last_price = self._get_last_price(ticker)
                        if last_price and last_price > 0:
                            whole_qty = max(1, int(notional // last_price))
                            return self._submit_order(ticker, side, notional=None, qty=whole_qty)
                    except Exception as price_exc:
                        logger.warning("No se pudo obtener precio de %s: %s", ticker, price_exc)

                # Error 429: rate limit → espera y reintenta
                if status_code == 429:
                    retry_after = float(
                        getattr(exc, "response", {}).get("Retry-After", backoff * 2)
                    ) if hasattr(exc, "response") else backoff * 2
                    logger.warning(
                        "Rate limit (intento %d/%d). Esperando %.1fs...",
                        attempt, _MAX_ORDER_RETRIES, retry_after,
                    )
                    time.sleep(retry_after)
                    backoff *= 2
                    continue

                # Error 5xx: servidor → backoff y reintento
                if status_code and status_code >= 500:
                    logger.warning(
                        "Error servidor %d para %s (intento %d/%d). Reintentando en %.1fs...",
                        status_code, ticker, attempt, _MAX_ORDER_RETRIES, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                # Error no recuperable
                logger.error("ORDEN FALLIDA | %-5s | %-10s | %s", action, ticker, exc)
                return OrderRecord(
                    timestamp=ts, ticker=ticker, action=action,
                    notional=notional or 0.0, qty=qty,
                    alpaca_order_id="", status="failed", error=str(exc),
                )

            except Exception as exc:
                logger.error("Error inesperado en orden %s %s: %s", action, ticker, exc)
                return OrderRecord(
                    timestamp=ts, ticker=ticker, action=action,
                    notional=notional or 0.0, qty=qty,
                    alpaca_order_id="", status="failed", error=str(exc),
                )

        logger.error("ORDEN FALLIDA tras %d intentos | %s %s", _MAX_ORDER_RETRIES, action, ticker)
        return OrderRecord(
            timestamp=ts, ticker=ticker, action=action,
            notional=notional or 0.0, qty=qty,
            alpaca_order_id="", status="failed",
            error=f"Agotados {_MAX_ORDER_RETRIES} reintentos",
        )

    def _close_position(self, ticker: str) -> OrderRecord:
        """
        Cierra la posición completa de un ticker usando el endpoint close_position.

        Alpaca cierra al precio de mercado con una sola llamada, independientemente
        del tamaño de la posición.
        """
        alpaca_symbol = _to_alpaca(ticker)
        ts = datetime.now(tz=timezone.utc).isoformat()
        try:
            order = self._client.close_position(symbol_or_asset_id=alpaca_symbol)
            record = OrderRecord(
                timestamp=ts, ticker=ticker, action="CLOSE",
                notional=0.0, qty=None,
                alpaca_order_id=order.id, status="submitted",
            )
            logger.info("CIERRE  POSIC | %-10s | id=%s", ticker, order.id)
            return record
        except APIError as exc:
            # Posición ya cerrada u otro error: registrar pero no explotar
            logger.warning("No se pudo cerrar %s: %s", ticker, exc)
            return OrderRecord(
                timestamp=ts, ticker=ticker, action="CLOSE",
                notional=0.0, qty=None,
                alpaca_order_id="", status="failed", error=str(exc),
            )

    def _get_last_price(self, ticker: str) -> float | None:
        """Obtiene el último precio de cierre disponible para un ticker."""
        try:
            request = StockLatestBarRequest(
                symbol_or_symbols=_to_alpaca(ticker),
                feed=ALPACA_FEED,
            )
            bars = self._data_client.get_stock_latest_bar(request)
            bar = bars.get(_to_alpaca(ticker))
            return float(bar.close) if bar else None
        except Exception as exc:
            logger.debug("No se pudo obtener precio de %s: %s", ticker, exc)
        return None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def execute(
        self,
        signals: list[SignalRecord],
        cancel_pending: bool = True,
    ) -> ExecutionSummary:
        """
        Ejecuta las señales del modelo: cierra posiciones obsoletas y abre nuevas.

        Flujo:
          1. Valida que el mercado esté abierto (advertencia si está cerrado).
          2. Cancela órdenes pendientes del día anterior (opcional).
          3. Lee estado actual de cuenta y posiciones.
          4. Selecciona hasta MAX_OPEN_POSITIONS señales de compra por confianza.
          5. Cierra posiciones que ya no tienen señal activa.
          6. Calcula notional por posición = equity / n_posiciones_objetivo.
          7. Abre posiciones nuevas.
          8. Registra toda la actividad en el log y devuelve el resumen.

        Args:
            signals: Lista de SignalRecord con ticker, signal y confidence.
            cancel_pending: Si True, cancela órdenes day pendientes antes de actuar.

        Returns:
            ExecutionSummary con contadores de operaciones y lista de órdenes.
        """
        ts_run = datetime.now(tz=timezone.utc).isoformat()
        logger.info("═══ Ejecución de señales [%s] ══════════════════════════", ts_run)

        # --- Validación de mercado ---
        if not self._is_market_open():
            logger.warning(
                "El mercado está CERRADO. Las órdenes 'day' se enviarán pero "
                "no se ejecutarán hasta la próxima apertura (o serán canceladas)."
            )

        # --- Cancelar órdenes pendientes ---
        if cancel_pending:
            try:
                cancelled = self._client.cancel_orders()
                if cancelled:
                    logger.info("Canceladas %d órdenes pendientes.", len(cancelled))
            except Exception as exc:
                logger.warning("No se pudieron cancelar órdenes pendientes: %s", exc)

        # --- Estado actual ---
        account = self._get_account()
        equity = account["equity"]
        buying_power = account["buying_power"]
        current_positions = self._get_open_positions()

        logger.info(
            "Cuenta: equity=$%.2f | buying_power=$%.2f | posiciones_abiertas=%d",
            equity, buying_power, len(current_positions),
        )

        # --- Selección de señales ---
        buy_signals = self._select_buy_signals(signals)
        buy_tickers = {s.ticker for s in buy_signals}
        all_signal_tickers = {s.ticker for s in signals}

        # --- Qué mantener vs cerrar ---
        to_close = [t for t in current_positions if t not in buy_tickers]
        to_open = [s for s in buy_signals if s.ticker not in current_positions]
        to_maintain = [t for t in current_positions if t in buy_tickers]

        logger.info(
            "Señales: %d recibidas | %d buy seleccionadas | "
            "%d abrir | %d cerrar | %d mantener",
            len(signals), len(buy_signals),
            len(to_open), len(to_close), len(to_maintain),
        )

        orders: list[OrderRecord] = []
        n_opened = 0
        n_closed = 0
        n_failed = 0

        # --- Paso 1: Cerrar posiciones obsoletas ---
        if to_close:
            logger.info("─── Cerrando %d posiciones obsoletas ──────────────", len(to_close))
        for ticker in to_close:
            record = self._close_position(ticker)
            orders.append(record)
            if record.status == "submitted":
                n_closed += 1
            else:
                n_failed += 1
            time.sleep(_ORDER_SLEEP)

        # --- Paso 2: Position sizing ---
        # Denominar sobre las posiciones que quedará abiertas tras los cierres
        n_target = len(to_maintain) + len(to_open)
        if n_target == 0:
            notional_per_pos = 0.0
        else:
            # Usamos buying_power actualizado (aproximado: no esperar settlements)
            # y repartimos entre todas las posiciones objetivo
            notional_per_pos = min(equity / n_target, buying_power / max(len(to_open), 1))

        logger.info(
            "Position sizing: equity=$%.2f / %d posiciones = $%.2f por posición.",
            equity, n_target, notional_per_pos,
        )

        # --- Paso 3: Abrir posiciones nuevas ---
        if to_open:
            logger.info("─── Abriendo %d posiciones nuevas ─────────────────", len(to_open))
        for sig in to_open:
            if notional_per_pos < self._min_notional:
                logger.warning(
                    "Notional $%.2f < mínimo $%.2f para %s. Omitiendo.",
                    notional_per_pos, self._min_notional, sig.ticker,
                )
                orders.append(OrderRecord(
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    ticker=sig.ticker, action="BUY",
                    notional=notional_per_pos, qty=None,
                    alpaca_order_id="", status="skipped",
                    error="notional_below_minimum",
                ))
                continue

            record = self._submit_order(sig.ticker, "buy", notional=notional_per_pos)
            orders.append(record)
            if record.status == "submitted":
                n_opened += 1
            else:
                n_failed += 1
            time.sleep(_ORDER_SLEEP)

        # --- Resumen ---
        summary = ExecutionSummary(
            timestamp=ts_run,
            portfolio_equity=equity,
            buying_power=buying_power,
            positions_before=len(current_positions),
            positions_after=n_target,
            signals_received=len(signals),
            buy_signals_selected=len(buy_signals),
            positions_opened=n_opened,
            positions_closed=n_closed,
            positions_maintained=len(to_maintain),
            orders_failed=n_failed,
            target_notional_per_position=notional_per_pos,
            orders=orders,
        )

        self._log_summary(summary)
        return summary

    def close_all(self) -> list[OrderRecord]:
        """
        Cierra todas las posiciones abiertas.

        Útil para fin de período de paper trading o para reset del portfolio.
        Cancela órdenes pendientes antes de cerrar posiciones.

        Returns:
            Lista de OrderRecord con el resultado de cada cierre.
        """
        logger.info("Cerrando TODAS las posiciones abiertas...")
        try:
            self._client.cancel_orders()
        except Exception as exc:
            logger.warning("No se pudieron cancelar órdenes pendientes: %s", exc)

        positions = self._get_open_positions()
        if not positions:
            logger.info("No hay posiciones abiertas.")
            return []

        records: list[OrderRecord] = []
        for ticker in positions:
            record = self._close_position(ticker)
            records.append(record)
            time.sleep(_ORDER_SLEEP)

        closed_ok = sum(1 for r in records if r.status == "submitted")
        logger.info("Cierres enviados: %d/%d.", closed_ok, len(records))
        return records

    # ------------------------------------------------------------------
    # Helpers de logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_summary(summary: ExecutionSummary) -> None:
        """Imprime el resumen de ejecución en el log."""
        logger.info("─── Resumen de ejecución ──────────────────────────────")
        logger.info("  Equity portfolio    : $%.2f", summary.portfolio_equity)
        logger.info("  Posiciones antes    : %d", summary.positions_before)
        logger.info("  Posiciones después  : %d (estimado)", summary.positions_after)
        logger.info("  Señales recibidas   : %d", summary.signals_received)
        logger.info("  Buy seleccionadas   : %d / max %d", summary.buy_signals_selected, MAX_OPEN_POSITIONS)
        logger.info("  Posiciones abiertas : %d", summary.positions_opened)
        logger.info("  Posiciones cerradas : %d", summary.positions_closed)
        logger.info("  Posiciones manten.  : %d", summary.positions_maintained)
        logger.info("  Órdenes fallidas    : %d", summary.orders_failed)
        logger.info("  Notional/posición   : $%.2f", summary.target_notional_per_position)
        logger.info("═══════════════════════════════════════════════════════")


# ---------------------------------------------------------------------------
# Adaptador desde DataFrame del modelo
# ---------------------------------------------------------------------------

def signals_from_dataframe(
    df: pd.DataFrame,
    signal_col: str = "signal",
    confidence_col: str = "confidence",
    ticker_col: str = "ticker",
) -> list[SignalRecord]:
    """
    Convierte un DataFrame de predicciones del modelo en lista de SignalRecord.

    El DataFrame debe tener una fila por ticker con las columnas indicadas.
    Si falta la columna de confianza se asume 0.5 para todos.

    Args:
        df: DataFrame con predicciones del modelo (una fila por ticker).
        signal_col: Nombre de la columna de señal binaria (0/1).
        confidence_col: Nombre de la columna de probabilidad de subida.
        ticker_col: Nombre de la columna de ticker.

    Returns:
        Lista de SignalRecord lista para AlpacaExecutor.execute().
    """
    records: list[SignalRecord] = []
    has_confidence = confidence_col in df.columns

    for _, row in df.iterrows():
        records.append(SignalRecord(
            ticker=str(row[ticker_col]),
            signal=int(row[signal_col]),
            confidence=float(row[confidence_col]) if has_confidence else 0.5,
        ))
    return records


# ---------------------------------------------------------------------------
# CLI de prueba
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    executor = AlpacaExecutor()
    account = executor._get_account()
    positions = executor._get_open_positions()

    print(f"\nEquity       : ${account['equity']:,.2f}")
    print(f"Buying power : ${account['buying_power']:,.2f}")
    print(f"Posiciones   : {len(positions)}")
    if positions:
        for ticker, qty in sorted(positions.items()):
            print(f"  {ticker:<12} qty={qty:.4f}")

    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        test_signals = [
            SignalRecord("AAPL", 1, 0.82),
            SignalRecord("MSFT", 1, 0.74),
            SignalRecord("NVDA", 1, 0.69),
            SignalRecord("TSLA", 0, 0.31),
        ]
        print(f"\nDry-run con {len(test_signals)} señales de prueba...")
        summary = executor.execute(test_signals)
        print(json.dumps(summary.to_dict(), indent=2, default=str))
