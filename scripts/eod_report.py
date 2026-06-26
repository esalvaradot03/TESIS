"""
Reporte End-of-Day de la cuenta de paper trading en Alpaca.

Genera un resumen de:
  - Equity, cash y buying power de la cuenta.
  - P&L del día (vía portfolio history) y P&L acumulado.
  - Posiciones abiertas con su P&L no realizado por símbolo.

Escribe el reporte como Markdown y JSON en data/live/reports/ para que el
workflow de GitHub Actions lo commitee y/o lo publique en el step summary.

Uso:
    python scripts/eod_report.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

REPORTS_DIR = DATA_DIR / "live" / "reports"


def _client():
    from alpaca.trading.client import TradingClient
    return TradingClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET, paper=True)


def gather_eod() -> dict:
    """Consulta cuenta, posiciones e historial de portfolio y arma el reporte."""
    from alpaca.trading.requests import GetPortfolioHistoryRequest

    client = _client()
    acc = client.get_account()
    positions = client.get_all_positions()

    # P&L del día vía portfolio history (resolución diaria)
    day_pl = None
    day_pl_pct = None
    try:
        hist = client.get_portfolio_history(
            GetPortfolioHistoryRequest(period="1D", timeframe="1H")
        )
        if hist.profit_loss:
            day_pl = float(hist.profit_loss[-1])
        if hist.profit_loss_pct:
            day_pl_pct = float(hist.profit_loss_pct[-1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo obtener portfolio history: %s", exc)

    pos_list = [
        {
            "symbol": p.symbol,
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price) if p.current_price else None,
            "market_value": float(p.market_value) if p.market_value else None,
            "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else None,
            "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else None,
        }
        for p in positions
    ]

    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "equity": float(acc.equity),
        "last_equity": float(acc.last_equity),
        "cash": float(acc.cash),
        "buying_power": float(acc.buying_power),
        "portfolio_value": float(acc.portfolio_value),
        "day_pl": day_pl,
        "day_pl_pct": day_pl_pct,
        "n_positions": len(pos_list),
        "positions": pos_list,
    }


def to_markdown(r: dict) -> str:
    """Formatea el reporte como Markdown."""
    lines = [
        f"# Reporte End-of-Day — {r['timestamp']}",
        "",
        "## Cuenta (paper)",
        f"- Equity: **${r['equity']:,.2f}** (equity previo: ${r['last_equity']:,.2f})",
        f"- Cash: ${r['cash']:,.2f} | Buying power: ${r['buying_power']:,.2f}",
    ]
    if r["day_pl"] is not None:
        sign = "🟢" if r["day_pl"] >= 0 else "🔴"
        pct = f" ({r['day_pl_pct']:+.2%})" if r["day_pl_pct"] is not None else ""
        lines.append(f"- P&L del día: {sign} ${r['day_pl']:+,.2f}{pct}")
    lines += ["", f"## Posiciones abiertas ({r['n_positions']})", ""]
    if r["positions"]:
        lines.append("| Símbolo | Qty | Entrada | Actual | Valor | P&L no realiz. |")
        lines.append("|---------|-----|---------|--------|-------|----------------|")
        for p in r["positions"]:
            upl = p["unrealized_pl"]
            upl_s = f"${upl:+,.2f}" if upl is not None else "—"
            cur = f"${p['current_price']:,.2f}" if p["current_price"] else "—"
            mv = f"${p['market_value']:,.2f}" if p["market_value"] else "—"
            lines.append(
                f"| {p['symbol']} | {p['qty']:.4f} | ${p['avg_entry_price']:,.2f} "
                f"| {cur} | {mv} | {upl_s} |"
            )
    else:
        lines.append("_Sin posiciones abiertas._")
    return "\n".join(lines) + "\n"


def main() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = gather_eod()
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d")

    md_path = REPORTS_DIR / f"eod_{ts}.md"
    json_path = REPORTS_DIR / f"eod_{ts}.json"
    md = to_markdown(report)
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(md)
    logger.info("Reporte EOD escrito en %s y %s", md_path, json_path)

    # Si corre en GitHub Actions, añadir al step summary
    summary_path = __import__("os").environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md)
    return md_path


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
