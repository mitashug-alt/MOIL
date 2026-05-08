"""Groq AI Macro Scoring Module

Handles automatic macro scoring using Groq LLM API (free tier) with proper error handling,
JSON parsing, and fallback mechanisms for the MOIL Macro Radar dashboard.
"""

import json
import os
from typing import Dict, Any, Optional, List

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    Groq = None

import pandas as pd
import streamlit as st


def get_groq_api_key() -> str:
    """Get the Groq API key from secrets or environment variables."""
    def get_secret(name: str) -> str:
        value = ""
        try:
            value = st.secrets.get(name, "")
        except Exception:
            value = ""
        if not value:
            value = os.getenv(name, "")
        return str(value or "").strip()

    return (
        get_secret("GROQ_API_KEY") or
        get_secret("API_KEY") or
        ""
    )


def get_groq_config() -> Dict[str, str]:
    """Get Groq metadata for UI display without exposing the API key."""
    def get_secret(name: str, default: str = "") -> str:
        value = ""
        try:
            value = st.secrets.get(name, "")
        except Exception:
            value = ""
        if not value:
            value = os.getenv(name, default)
        return str(value or "").strip()

    if get_secret("GROQ_API_KEY"):
        key_name = "GROQ_API_KEY"
    elif get_secret("API_KEY"):
        key_name = "API_KEY"
    else:
        key_name = "GROQ_API_KEY"

    return {
        "model": get_secret("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "project": get_secret("GROQ_PROJECT", ""),
        "key_name": key_name,
    }


def is_groq_configured() -> bool:
    """Check if Groq (Groq) is properly configured."""
    return bool(get_groq_api_key() and HAS_GROQ)


def call_groq_json(prompt: str, model: str = "llama-3.3-70b-versatile") -> Optional[Dict[str, Any]]:
    """Call Groq LLM API and return parsed JSON response."""
    if not is_groq_configured():
        return None

    try:
        api_key = get_groq_api_key()
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a JSON-only API. Always respond with valid JSON, no markdown fences, no extra text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        if response and response.choices:
            text = response.choices[0].message.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            return json.loads(text)
        else:
            st.error("Empty response from Groq API")
            return None

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse Groq JSON response: {e}")
        return None
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return None


def score_macro_with_groq(
    manual_macro_df: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    regime_context: Dict[str, Any],
    news_tracker_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    """Generate comprehensive macro scoring using Groq AI."""

    if not manual_macro_df.empty:
        macro_indicators = manual_macro_df.to_dict('records')
    else:
        macro_indicators = []

    # Format news tracker for context
    news_context = ""
    if news_tracker_df is not None and not news_tracker_df.empty:
        news_context = "\nNEWS & INSTITUTIONAL ACTIVITY:\n" + "\n".join([
            f"- {row['item']} ({row['category']}): {row['value']} (Impact: {row['impact']}) - {row['details']}"
            for _, row in news_tracker_df.iterrows()
        ])

    # Format market snapshot for context
    market_context = ""
    if not market_snapshot.empty:
        market_context = "\n".join([
            f"- {row['asset']}: {row['latest']:.2f} ({row['1D %']:+.2%})"
            for _, row in market_snapshot.iterrows()
            if pd.notna(row['1D %'])
        ])

    prompt = f"""You are a senior commodities analyst at an institutional investment firm specializing in industrial metals and steel cycle analysis.

Your task is to analyze the current macro environment for MOIL Ltd (Indian manganese ore producer) and provide quantitative scores for key macro indicators.

CURRENT MARKET CONTEXT:
{market_context}

CURRENT REGIME: {regime_context.get('label', 'Unknown')} (Score: {regime_context.get('normalized_score', 50):.1f}/100)
{news_context}

MACRO INDICATORS TO SCORE:
{json.dumps(macro_indicators, indent=2)}

Please analyze each macro indicator and provide:

1. **macro_scores**: Array of objects with:
   - indicator: The indicator name
   - score: Numeric score from -2 (very bearish) to +2 (very bullish)
   - status: One of ["bullish", "constructive", "neutral", "watch", "bearish"]
   - commentary: 1-2 sentence explanation

2. **overall_manual_macro_score**: Single numeric score from -2 to +2 representing the net macro impact

3. **regime_commentary**: 2-3 paragraph analysis of current regime and macro drivers

4. **watch_items**: Array of 2-3 key items requiring immediate attention

5. **risks**: Array of 2-3 major downside risks to monitor

Consider MOIL's exposure to:
- Indian steel demand (domestic consumption)
- China steel exports (global oversupply risk)
- Silico-manganese prices (direct product pricing)
- Energy costs (power stress in production)
- Freight rates (export competitiveness)
- USDINR (currency impact on exports)
- Institutional buy/sell activity (FII/DII sentiment)
- Recent fund deratings or negative news flow

Be quantitative and specific in your analysis. Focus on actionable insights for commodity trading desks.

Return your analysis as a valid JSON object with the exact structure specified above."""

    result = call_groq_json(prompt)

    if result:
        # Validate the response structure
        required_keys = ["macro_scores", "overall_manual_macro_score", "regime_commentary", "watch_items", "risks"]
        if all(key in result for key in required_keys):
            return result
        else:
            st.error(f"Groq response missing required keys. Got: {list(result.keys())}")
            return None
    else:
        return None





def extract_news_updates(raw_news_text: str) -> Optional[List[Dict[str, Any]]]:
    """Use Groq to extract structured fund activity and news from raw text."""
    if not is_groq_configured() or not raw_news_text.strip():
        return None

    prompt = f"""
You are an institutional data extractor. Extract mutual fund holdings, institutional investor actions (buy/sell/exit), and negative news/deratings for MOIL Ltd from the raw text below.

RAW TEXT:
{raw_news_text}

RULES:
- Extract specific fund names.
- Identify the category: "Mutual Fund", "Institutional", "Fund Action", or "News / Deratings".
- Identify the value (e.g., stake %, "Exited", "Bought").
- Estimate impact: -2.0 (very negative) to +2.0 (very positive).
- Provide context in the "details" field.
- Return ONLY a JSON list of objects.

JSON STRUCTURE:
[
  {{"category": "Mutual Fund", "item": "Fund Name", "value": "0.5%", "impact": 0.5, "details": "context"}},
  ...
]
"""
    result = call_groq_json(prompt)
    if isinstance(result, list):
        return result
    elif isinstance(result, dict) and "updates" in result:
        return result["updates"]
    return None


def generate_groq_commentary(context: dict | None = None, **kwargs) -> dict:
    """
    Generate Groq commentary for MOIL Macro Radar using Groq LLM.

    Returns a safe dictionary:
    {
        "ok": bool,
        "telegram_alert": str,
        "institutional_commentary": str,
        "watchlist_triggers": list,
        "raw_text": str,
        "error": str
    }

    This function never displays or logs the API key.
    """
    import json
    import os
    import re

    try:
        import streamlit as st
    except Exception:
        st = None

    try:
        from groq import Groq
    except Exception as exc:
        return {
            "ok": False,
            "telegram_alert": "",
            "institutional_commentary": "",
            "watchlist_triggers": [],
            "raw_text": "",
            "error": f"groq is not installed or could not be imported: {exc}",
        }

    def get_secret(name: str, default: str = "") -> str:
        value = ""

        if st is not None:
            try:
                value = st.secrets.get(name, "")
            except Exception:
                value = ""

        if not value:
            value = os.getenv(name, default)

        return str(value or "").strip()

    def strip_json_fences(text: str) -> str:
        text = (text or "").strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]

        return text

    context = context or {}
    if kwargs:
        context.update(kwargs)

    # Include news tracker in context if provided
    if "news_tracker" in context:
        news_df = context["news_tracker"]
        if isinstance(news_df, pd.DataFrame) and not news_df.empty:
            context["news_summary"] = news_df.to_dict('records')
            # Don't pass the raw dataframe to json.dumps later
            del context["news_tracker"]

    api_key = get_secret("GROQ_API_KEY")
    model = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        return {
            "ok": False,
            "telegram_alert": "",
            "institutional_commentary": "",
            "watchlist_triggers": [],
            "raw_text": "",
            "error": "Groq API key is not configured. Set GROQ_API_KEY in Streamlit secrets or environment variables.",
        }

    prompt = f"""
You are an institutional commodity and equity-cycle analyst writing for the MOIL Macro Radar.

Task:
Generate macro commentary for MOIL Ltd using only the dashboard context below.

Rules:
- Do not give direct buy/sell advice.
- Do not hallucinate missing values.
- Be concise, institutional, and risk-aware.
- Focus on manganese, silico-manganese, ferroalloys, Indian steel demand, China exports, Brent, USDINR, freight, and commodity-cycle regime.
- Return valid JSON only.
- No markdown fences.

Dashboard context:
{json.dumps(context, indent=2, default=str)}

Return this exact JSON structure:
{{
  "telegram_alert": "Telegram-ready alert under 150 words",
  "institutional_commentary": "Longer institutional commentary in 2-4 paragraphs",
  "watchlist_triggers": [
    "Trigger 1",
    "Trigger 2",
    "Trigger 3"
  ]
}}
"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a JSON-only API. Always respond with valid JSON, no markdown fences, no extra text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content if response.choices else ""
        clean_text = strip_json_fences(raw_text)

        try:
            parsed = json.loads(clean_text)
        except Exception:
            return {
                "ok": False,
                "telegram_alert": "",
                "institutional_commentary": "",
                "watchlist_triggers": [],
                "raw_text": raw_text,
                "error": "Groq returned non-JSON text. Raw response captured.",
            }

        return {
            "ok": True,
            "telegram_alert": str(parsed.get("telegram_alert", "")).strip(),
            "institutional_commentary": str(parsed.get("institutional_commentary", "")).strip(),
            "watchlist_triggers": parsed.get("watchlist_triggers", []),
            "raw_text": raw_text,
            "error": "",
        }

    except Exception as exc:
        return {
            "ok": False,
            "telegram_alert": "",
            "institutional_commentary": "",
            "watchlist_triggers": [],
            "raw_text": "",
            "error": f"Groq commentary failed: {exc}",
        }