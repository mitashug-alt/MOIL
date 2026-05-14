from macro_radar import (
    DEFAULT_MARKET_TICKERS,
    build_price_panel,
    compute_regime_score,
    correlation_matrix,
    detect_return_anomalies,
    generate_demo_market_data,
    generate_rule_based_commentary,
    moil_correlation_table,
)
import pandas as pd


def main() -> None:
    data = generate_demo_market_data(DEFAULT_MARKET_TICKERS.keys(), periods=260)
    prices = build_price_panel(data)
    corr = correlation_matrix(prices, lookback=90)
    moil_corr = moil_correlation_table(corr)
    anomalies = detect_return_anomalies(prices)
    # Create minimal manual_macro DataFrame for smoke test
    manual_macro = pd.DataFrame({
        "date": ["2026-05-12"],
        "indicator": ["Silico-Manganese Prices"],
        "value": [""],
        "unit": ["INR/t"],
        "status": ["neutral"],
        "score": [0.0],
        "commentary": ["Smoke test"],
        "source": ["Test"],
    })
    scorecard, summary = compute_regime_score(prices, manual_macro)
    commentary = generate_rule_based_commentary(summary, scorecard, moil_corr, anomalies)

    assert not prices.empty
    assert "MOIL" in prices.columns
    assert not scorecard.empty
    assert 0 <= summary.normalized_score <= 100
    assert "Regime:" in commentary
    print("Smoke test passed")


if __name__ == "__main__":
    main()
