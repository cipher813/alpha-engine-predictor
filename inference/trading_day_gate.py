"""Trading-day gate for the weekday Step Function.

Exposed as ``action=check_trading_day`` on the predictor Lambda handler so the
weekday pipeline can decide whether to start the executor BEFORE booting the
trading box — replacing the prior on-box SSM ``trading_calendar`` check, whose
stdout was unreliably captured on a just-cold-booted instance (2026-06-30 false
TradingDayCheckFailed alert; config#1430).

Pure NYSE-calendar math via ``nousergon_lib.trading_calendar`` — no S3, no
ArcticDB, no models, no GitHub. The gate must never depend on fragile infra, so
that the gate itself can't be the thing that breaks.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

_NYSE = ZoneInfo("America/New_York")


def check_trading_day(date_str: str | None = None) -> dict:
    """Return whether ``date_str`` (or today in NYSE time) is an NYSE trading day."""
    from nousergon_lib.trading_calendar import is_trading_day, next_trading_day

    check_date = date.fromisoformat(date_str) if date_str else datetime.now(_NYSE).date()
    trading = bool(is_trading_day(check_date))
    result = {
        "is_trading_day": trading,
        "check_date": check_date.isoformat(),
        "day_name": check_date.strftime("%A"),
        "marker": "TRADING DAY" if trading else "MARKET_CLOSED",
    }
    if not trading:
        result["reason"] = "weekend" if check_date.weekday() > 4 else "NYSE holiday"
        result["next_trading_day"] = next_trading_day(check_date).isoformat()
    return result
