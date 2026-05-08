# Future feed connectors

This folder is reserved for licensed or internal connectors.

Recommended pattern:

1. Each connector writes a normalized CSV or Parquet file into `data/`.
2. Required columns: `date`, `indicator`, `value`, `unit`, `status`, `score`, `commentary`, `source`.
3. Keep proprietary Reuters / SteelMint credentials out of the repository.
4. Convert raw feed direction into a bounded score from -2 to +2 before the Streamlit app consumes it.

Suggested connector modules:

- `reuters_china_industry.py`
- `steelmint_manganese_prices.py`
- `india_steel_production.py`
- `china_steel_exports.py`
- `power_stress_monitor.py`
