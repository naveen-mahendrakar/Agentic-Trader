from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP

server = FastMCP("technical-analysis")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [col[0] if isinstance(col, tuple) else col for col in normalized.columns]
    return normalized


def _download_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f"No market data returned for {symbol}")

    data = _normalize_columns(data)
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(set(data.columns)):
        raise ValueError(f"Expected OHLCV columns for {symbol}, got {list(data.columns)}")

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data


def _compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Close"])
    frame["SMA20"] = frame["Close"].rolling(window=20).mean()
    frame["SMA50"] = frame["Close"].rolling(window=50).mean()
    frame["SMA100"] = frame["Close"].rolling(window=100).mean()
    frame["SMA200"] = frame["Close"].rolling(window=200).mean()

    delta = frame["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    frame["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

    ema12 = frame["Close"].ewm(span=12, adjust=False).mean()
    ema26 = frame["Close"].ewm(span=26, adjust=False).mean()
    frame["MACD"] = ema12 - ema26
    frame["MACD_SIGNAL"] = frame["MACD"].ewm(span=9, adjust=False).mean()
    frame["MACD_HIST"] = frame["MACD"] - frame["MACD_SIGNAL"]

    typical_price = (frame["High"] + frame["Low"] + frame["Close"]) / 3
    frame["BB_MIDDLE"] = frame["Close"].rolling(window=20).mean()
    std = frame["Close"].rolling(window=20).std()
    frame["BB_UPPER"] = frame["BB_MIDDLE"] + (2 * std)
    frame["BB_LOWER"] = frame["BB_MIDDLE"] - (2 * std)

    frame["ATR"] = (
        pd.concat(
            [
                frame["High"] - frame["Low"],
                (frame["High"] - frame["Close"].shift()).abs(),
                (frame["Low"] - frame["Close"].shift()).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(window=14)
        .mean()
    )

    frame["VOL_MA20"] = pd.to_numeric(frame["Volume"], errors="coerce").rolling(window=20).mean()
    frame["VOL_MA50"] = pd.to_numeric(frame["Volume"], errors="coerce").rolling(window=50).mean()
    frame["TRIGGER"] = frame["Close"] > frame["SMA20"]
    return frame


def _summarize_signal(frame: pd.DataFrame) -> dict[str, Any]:
    latest = frame.iloc[-1]
    prev_close = frame["Close"].iloc[-2] if len(frame) > 1 else latest["Close"]
    change_pct = ((latest["Close"] - prev_close) / prev_close * 100) if prev_close else 0.0

    rsi = float(latest["RSI"])
    if rsi >= 70:
        rsi_signal = "overbought"
    elif rsi <= 30:
        rsi_signal = "oversold"
    else:
        rsi_signal = "neutral"

    price = float(latest["Close"])
    sma20 = float(latest["SMA20"])
    sma50 = float(latest["SMA50"])
    sma200 = float(latest["SMA200"])

    if price > sma20 and price > sma50 and price > sma200:
        trend = "bullish"
    elif price < sma20 and price < sma50 and price < sma200:
        trend = "bearish"
    else:
        trend = "mixed"

    macd = float(latest["MACD"])
    signal = float(latest["MACD_SIGNAL"])
    momentum = "bullish" if macd > signal else "bearish"

    recent_high = float(frame["Close"].tail(20).max())
    recent_low = float(frame["Close"].tail(20).min())
    return {
        "trend": trend,
        "momentum": momentum,
        "rsi": round(rsi, 2),
        "rsi_signal": rsi_signal,
        "change_pct": round(change_pct, 2),
        "support": round(recent_low, 2),
        "resistance": round(recent_high, 2),
        "price": round(price, 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "macd": round(macd, 3),
        "macd_signal": round(signal, 3),
    }


@server.tool()
def analyze_stock(symbol: str, period: str = "1y", interval: str = "1d") -> dict[str, Any]:
    """Run a technical analysis workflow for a stock symbol and return a JSON summary."""
    data = _download_history(symbol, period=period, interval=interval)
    frame = _compute_indicators(data)
    summary = _summarize_signal(frame)
    latest = frame.iloc[-1]
    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "latest_close": round(float(latest["Close"]), 2),
        "latest_rsi": round(float(latest["RSI"]), 2),
        "latest_macd": round(float(latest["MACD"]), 3),
        "year_high": round(float(frame["Close"].max()), 2),
        "year_low": round(float(frame["Close"].min()), 2),
    }


@server.tool()
def create_chart(symbol: str, period: str = "1y", interval: str = "1d", output_dir: str = "reports/technical_analysis") -> dict[str, Any]:
    """Create a PNG chart for the symbol's technical analysis and return its path."""
    data = _download_history(symbol, period=period, interval=interval)
    frame = _compute_indicators(data)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_path = output_path / f"{symbol.replace('.', '_')}_{period}_{interval}.png"

    fig, (ax_price, ax_volume) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax_price.plot(frame.index, frame["Close"], label="Close", color="#1f77b4", linewidth=1.5)
    ax_price.plot(frame.index, frame["SMA20"], label="SMA20", color="#ff7f0e", linewidth=1.0)
    ax_price.plot(frame.index, frame["SMA50"], label="SMA50", color="#2ca02c", linewidth=1.0)
    ax_price.plot(frame.index, frame["SMA200"], label="SMA200", color="#d62728", linewidth=1.0)
    ax_price.fill_between(frame.index, frame["BB_UPPER"], frame["BB_LOWER"], color="silver", alpha=0.25, label="Bollinger Band")
    ax_price.set_title(f"{symbol.upper()} Technical Analysis")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.3)

    ax_volume.bar(frame.index, frame["Volume"], color="#7f7f7f", alpha=0.6)
    ax_volume.set_ylabel("Volume")
    ax_volume.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "symbol": symbol.upper(),
        "chart_path": str(chart_path),
        "chart_exists": chart_path.exists(),
    }


@server.tool()
def generate_html_report(symbol: str, period: str = "1y", interval: str = "1d", output_dir: str = "reports/technical_analysis") -> dict[str, Any]:
    """Generate an HTML technical analysis report with chart and key metrics."""
    analysis = analyze_stock(symbol=symbol, period=period, interval=interval)
    chart_result = create_chart(symbol=symbol, period=period, interval=interval, output_dir=output_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"{symbol.replace('.', '_')}_{period}_{interval}.html"

    chart_name = Path(chart_result["chart_path"]).name
    summary = analysis["summary"]
    html_content = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>{html.escape(symbol.upper())} Technical Analysis Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
        h1, h2 {{ color: #111827; }}
        .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
        th {{ background: #f3f4f6; }}
        img {{ max-width: 100%; border: 1px solid #d1d5db; border-radius: 6px; }}
      </style>
    </head>
    <body>
      <h1>{html.escape(symbol.upper())} Technical Analysis Report</h1>
      <p><strong>Period:</strong> {html.escape(period)} &nbsp; <strong>Interval:</strong> {html.escape(interval)}</p>
      <div class="card">
        <h2>Summary</h2>
        <ul>
          <li><strong>Trend:</strong> {html.escape(summary['trend'])}</li>
          <li><strong>Momentum:</strong> {html.escape(summary['momentum'])}</li>
          <li><strong>RSI:</strong> {summary['rsi']} ({html.escape(summary['rsi_signal'])})</li>
          <li><strong>Price:</strong> {summary['price']}</li>
          <li><strong>Support:</strong> {summary['support']} | <strong>Resistance:</strong> {summary['resistance']}</li>
        </ul>
      </div>
      <div class="card">
        <h2>Chart</h2>
        <img src="{html.escape(chart_name)}" alt="{html.escape(symbol.upper())} technical chart">
      </div>
      <div class="card">
        <h2>Key Metrics</h2>
        <table>
          <tr><th>Metric</th><th>Value</th></tr>
          <tr><td>Latest Close</td><td>{analysis['latest_close']}</td></tr>
          <tr><td>RSI</td><td>{analysis['latest_rsi']}</td></tr>
          <tr><td>MACD</td><td>{analysis['latest_macd']}</td></tr>
          <tr><td>Year High</td><td>{analysis['year_high']}</td></tr>
          <tr><td>Year Low</td><td>{analysis['year_low']}</td></tr>
        </table>
      </div>
    </body>
    </html>
    """

    report_path.write_text(html_content, encoding="utf-8")
    return {
        "symbol": symbol.upper(),
        "report_path": str(report_path),
        "chart_path": chart_result["chart_path"],
        "report_exists": report_path.exists(),
    }


if __name__ == "__main__":
    server.run()
