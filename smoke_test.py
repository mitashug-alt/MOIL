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


def main() -> None:
    data = generate_demo_market_data(DEFAULT_MARKET_TICKERS.keys(), periods=260)
    prices = build_price_panel(data)
    corr = correlation_matrix(prices, lookback=90)
    moil_corr = moil_correlation_table(corr)
    anomalies = detect_return_anomalies(prices)
    scorecard, summary = compute_regime_score(prices)
    commentary = generate_rule_based_commentary(summary, scorecard, moil_corr, anomalies)

    assert not prices.empty
    assert "MOIL" in prices.columns
    assert not scorecard.empty
    assert 0 <= summary.normalized_score <= 100
    assert "Regime read-through" in commentary
    print("Smoke test passed")


if __name__ == "__main__":
    main()
