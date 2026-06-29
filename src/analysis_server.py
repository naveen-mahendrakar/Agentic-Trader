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

server = FastMCP("analysis")


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


def _fetch_fundamental_data(symbol: str) -> dict[str, Any]:
    """Fetch fundamental analysis parameters for a stock."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        fundamentals = {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "eps": info.get("trailingEps"),
            "book_value": info.get("bookValue"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "dividend_yield": info.get("dividendYield"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "company_name": info.get("longName", symbol.upper()),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
        
        # Format values for display
        formatted = {}
        for key, value in fundamentals.items():
            if value is not None:
                if key in ["pe_ratio", "pb_ratio", "roe", "roa", "eps", "book_value", "debt_to_equity", "current_ratio", "dividend_yield", "profit_margin", "operating_margin", "fifty_two_week_high", "fifty_two_week_low"]:
                    formatted[key] = round(float(value), 2) if isinstance(value, (int, float)) else value
                else:
                    formatted[key] = value
            else:
                formatted[key] = "N/A"
        
        return formatted
    except Exception as e:
        return {"error": str(e)}


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
    """Run a technical and fundamental analysis workflow for a stock symbol and return a JSON summary."""
    data = _download_history(symbol, period=period, interval=interval)
    frame = _compute_indicators(data)
    summary = _summarize_signal(frame)
    fundamentals = _fetch_fundamental_data(symbol)
    latest = frame.iloc[-1]
    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "fundamentals": fundamentals,
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
    """Generate a comprehensive HTML analysis report with technical and fundamental analysis."""
    analysis = analyze_stock(symbol=symbol, period=period, interval=interval)
    chart_result = create_chart(symbol=symbol, period=period, interval=interval, output_dir=output_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"{symbol.replace('.', '_')}_{period}_{interval}.html"

    chart_name = Path(chart_result["chart_path"]).name
    summary = analysis["summary"]
    fundamentals = analysis.get("fundamentals", {})
    
    # Build fundamental analysis section
    fundamental_rows = ""
    if fundamentals and "error" not in fundamentals:
        fund_metrics = [
            ("Company", "company_name"),
            ("Sector", "sector"),
            ("Industry", "industry"),
            ("Market Cap", "market_cap"),
            ("P/E Ratio", "pe_ratio"),
            ("P/B Ratio", "pb_ratio"),
            ("EPS", "eps"),
            ("ROE", "roe"),
            ("ROA", "roa"),
            ("Debt-to-Equity", "debt_to_equity"),
            ("Current Ratio", "current_ratio"),
            ("Dividend Yield", "dividend_yield"),
            ("Profit Margin", "profit_margin"),
            ("Operating Margin", "operating_margin"),
            ("52-Week High", "fifty_two_week_high"),
            ("52-Week Low", "fifty_two_week_low"),
        ]
        for label, key in fund_metrics:
            value = fundamentals.get(key, "N/A")
            fundamental_rows += f"<tr><td><strong>{label}</strong></td><td>{html.escape(str(value))}</td></tr>\n"
    else:
        fundamental_rows = "<tr><td colspan='2'>Fundamental data not available</td></tr>"

    html_content = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>{html.escape(symbol.upper())} Analysis Report</title>
      <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 24px; color: #1f2937; background: #f9fafb; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #111827; border-bottom: 3px solid #3b82f6; padding-bottom: 12px; }}
        h2 {{ color: #1f2937; margin-top: 24px; margin-bottom: 12px; }}
        .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 16px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 10px 12px; text-align: left; }}
        th {{ background: #f3f4f6; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f9fafb; }}
        img {{ max-width: 100%; border: 1px solid #d1d5db; border-radius: 6px; }}
        .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 16px; }}
        .summary-item {{ background: #f3f4f6; padding: 12px; border-radius: 6px; border-left: 4px solid #3b82f6; }}
        .summary-item strong {{ display: block; color: #6b7280; font-size: 12px; }}
        .summary-item span {{ display: block; font-size: 18px; font-weight: 700; color: #111827; }}
        .bullish {{ color: #10b981; font-weight: 600; }}
        .bearish {{ color: #ef4444; font-weight: 600; }}
        .neutral {{ color: #f59e0b; font-weight: 600; }}
        .technical-section {{ background: #eff6ff; border-left: 4px solid #3b82f6; }}
        .fundamental-section {{ background: #f0fdf4; border-left: 4px solid #10b981; }}
        ul {{ margin: 8px 0; padding-left: 20px; }}
        li {{ margin: 6px 0; }}
      </style>
    </head>
    <body>
      <div class="container">
        <h1>{html.escape(symbol.upper())} - Technical & Fundamental Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} &nbsp; <strong>Period:</strong> {html.escape(period)} &nbsp; <strong>Interval:</strong> {html.escape(interval)}</p>
        
        <div class="card technical-section">
          <h2>Technical Analysis Summary</h2>
          <div class="summary-grid">
            <div class="summary-item">
              <strong>TREND</strong>
              <span class="{html.escape(summary['trend'])}">{html.escape(summary['trend']).upper()}</span>
            </div>
            <div class="summary-item">
              <strong>MOMENTUM</strong>
              <span class="{html.escape(summary['momentum'])}">{html.escape(summary['momentum']).upper()}</span>
            </div>
            <div class="summary-item">
              <strong>RSI</strong>
              <span>{summary['rsi']} ({html.escape(summary['rsi_signal'])})</span>
            </div>
          </div>
          <ul>
            <li><strong>Current Price:</strong> {summary['price']}</li>
            <li><strong>Price Change (24h):</strong> {summary['change_pct']}%</li>
            <li><strong>Support Level:</strong> {summary['support']} | <strong>Resistance Level:</strong> {summary['resistance']}</li>
            <li><strong>Moving Averages:</strong> SMA20: {summary['sma20']}, SMA50: {summary['sma50']}, SMA200: {summary['sma200']}</li>
            <li><strong>MACD:</strong> {summary['macd']} (Signal: {summary['macd_signal']})</li>
          </ul>
        </div>

        <div class="card">
          <h2>Price Chart</h2>
          <img src="{html.escape(chart_name)}" alt="{html.escape(symbol.upper())} technical chart">
        </div>

        <div class="grid">
          <div class="card">
            <h2>Technical Metrics</h2>
            <table>
              <tr><th>Metric</th><th>Value</th></tr>
              <tr><td>Latest Close</td><td>{analysis['latest_close']}</td></tr>
              <tr><td>RSI (14)</td><td>{analysis['latest_rsi']}</td></tr>
              <tr><td>MACD</td><td>{analysis['latest_macd']}</td></tr>
              <tr><td>52-Week High</td><td>{analysis['year_high']}</td></tr>
              <tr><td>52-Week Low</td><td>{analysis['year_low']}</td></tr>
            </table>
          </div>

          <div class="card fundamental-section">
            <h2>Fundamental Metrics</h2>
            <table>
              {fundamental_rows}
            </table>
          </div>
        </div>

        <div class="card">
          <h2>Key Insights</h2>
          <ul>
            <li><strong>Technical Status:</strong> The stock shows a <strong>{html.escape(summary['trend'])}</strong> trend with <strong>{html.escape(summary['momentum'])}</strong> momentum. RSI reading of <strong>{summary['rsi']}</strong> indicates the asset is <strong>{html.escape(summary['rsi_signal'])}</strong>.</li>
            <li><strong>Price Action:</strong> Trading {"above" if summary['price'] > summary['sma200'] else "below"} the 200-day SMA, suggesting {"strength" if summary['price'] > summary['sma200'] else "weakness"} in the longer-term trend.</li>
            <li><strong>Valuation:</strong> {"P/E ratio and other valuation metrics are available above for fundamental assessment." if fundamentals.get('pe_ratio', 'N/A') != 'N/A' else "Fundamental data is limited; check with official sources for comprehensive analysis."}</li>
            <li><strong>Risk Levels:</strong> Support is at {summary['support']}, Resistance is at {summary['resistance']}. Use these levels for risk management.</li>
          </ul>
        </div>

        <div class="card" style="background: #fef2f2; border-left: 4px solid #ef4444;">
          <h2>Disclaimer</h2>
          <p>This report is for informational purposes only and should not be considered as financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions. Past performance is not indicative of future results.</p>
        </div>
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
