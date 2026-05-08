"""Gemini AI Macro Scoring Module

Handles automatic macro scoring using Google Gemini AI with proper error handling,
JSON parsing, and fallback mechanisms for the MOIL Macro Radar dashboard.
"""

import json
import os
from typing import Dict, Any, Optional, List

try:
    import google.genai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None

import pandas as pd
import streamlit as st


def get_gemini_api_key() -> str:
    """Get the Gemini API key from secrets or environment variables."""
    return (
        st.secrets.get("GEMINI_API_KEY") or
        st.secrets.get("GOOGLE_API_KEY") or
        st.secrets.get("API_KEY") or
        os.getenv("GEMINI_API_KEY") or
        os.getenv("GOOGLE_API_KEY") or
        ""
    )


def get_gemini_config() -> Dict[str, str]:
    """Get Gemini metadata for UI display without exposing the API key."""
    if st.secrets.get("GEMINI_API_KEY"):
        key_name = "GEMINI_API_KEY"
    elif st.secrets.get("GOOGLE_API_KEY"):
        key_name = "GOOGLE_API_KEY"
    elif st.secrets.get("API_KEY"):
        key_name = "API_KEY"
    else:
        key_name = "GEMINI_API_KEY"

    return {
        "model": st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash"),
        "project": st.secrets.get("GEMINI_PROJECT", ""),
        "key_name": key_name,
    }


def is_gemini_configured() -> bool:
    """Check if Gemini is properly configured."""
    return bool(get_gemini_api_key() and HAS_GENAI)


def call_gemini_json(prompt: str, model: str = "gemini-1.5-flash") -> Optional[Dict[str, Any]]:
    """Call Gemini API and return parsed JSON response."""
    if not is_gemini_configured():
        return None

    try:
        api_key = get_gemini_api_key()
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,  # Low temperature for consistent scoring
            )
        )

        if response and response.text:
            # Clean the response text
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            # Parse JSON
            return json.loads(text)
        else:
            st.error("Empty response from Gemini API")
            return None

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse Gemini JSON response: {e}")
        return None
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None


def score_macro_with_gemini(
    manual_macro_df: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    regime_context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Generate comprehensive macro scoring using Gemini AI."""

    if not manual_macro_df.empty:
        macro_indicators = manual_macro_df.to_dict('records')
    else:
        macro_indicators = []

    # Format market snapshot for context
    market_context = ""
    if not market_snapshot.empty:
        market_context = "\n".join([
            f"- {row['Asset']}: {row['Close']:.2f} ({row['1D %']:+.2%})"
            for _, row in market_snapshot.iterrows()
            if pd.notna(row['1D %'])
        ])

    prompt = f"""You are a senior commodities analyst at an institutional investment firm specializing in industrial metals and steel cycle analysis.

Your task is to analyze the current macro environment for MOIL Ltd (Indian manganese ore producer) and provide quantitative scores for key macro indicators.

CURRENT MARKET CONTEXT:
{market_context}

CURRENT REGIME: {regime_context.get('label', 'Unknown')} (Score: {regime_context.get('normalized_score', 50):.1f}/100)

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

Be quantitative and specific in your analysis. Focus on actionable insights for commodity trading desks.

Return your analysis as a valid JSON object with the exact structure specified above."""

    result = call_gemini_json(prompt)

    if result:
        # Validate the response structure
        required_keys = ["macro_scores", "overall_manual_macro_score", "regime_commentary", "watch_items", "risks"]
        if all(key in result for key in required_keys):
            return result
        else:
            st.error(f"Gemini response missing required keys. Got: {list(result.keys())}")
            return None
    else:
        return None


def generate_gemini_commentary(context: Dict[str, Any]) -> Optional[str]:
    """Generate AI commentary using Gemini."""

    prompt = f"""You are a commodities desk analyst writing a concise macro alert for MOIL Ltd.

CURRENT REGIME: {context.get('regime_label', 'Unknown')} (Score: {context.get('regime_score', 50):.1f}/100)
MANUAL MACRO SCORE: {context.get('manual_macro_score', 0):+.2f}

TOP CORRELATIONS:
{chr(10).join([f"- {item['asset']}: {item['correlation_to_moil']:+.2f}" for item in context.get('top_correlations', [])])}

ANOMALIES DETECTED:
{chr(10).join([f"- {item['asset']}: {item['latest_return_%']:+.2%} (z-score: {item['z_score']:+.2f})" for item in context.get('anomalies', [])])}

Write a 2-3 paragraph dashboard note covering:
1. Current regime assessment and momentum
2. Key supportive and adverse signals from correlations and anomalies
3. Top risks and recommended positioning for MOIL exposure

Keep it concise, quantitative, and focused on actionable insights for institutional investors."""

    if not is_gemini_configured():
        return None

    try:
        api_key = get_gemini_api_key()
        client = genai.Client(api_key=api_key)

        config = get_gemini_config()
        response = client.models.generate_content(
            model=config["model"],
            contents=prompt,
            config=genai.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1000,
            )
        )

        if response and response.text:
            return response.text.strip()
        else:
            return None

    except Exception as e:
        st.error(f"Gemini commentary error: {e}")
        return None


def generate_gemini_commentary(context: dict | None = None, **kwargs) -> dict:
    """
    Generate Gemini commentary for MOIL Macro Radar.

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
        from google import genai
    except Exception as exc:
        return {
            "ok": False,
            "telegram_alert": "",
            "institutional_commentary": "",
            "watchlist_triggers": [],
            "raw_text": "",
            "error": f"google-genai is not installed or could not be imported: {exc}",
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

    api_key = get_secret("GEMINI_API_KEY")
    model = get_secret("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        return {
            "ok": False,
            "telegram_alert": "",
            "institutional_commentary": "",
            "watchlist_triggers": [],
            "raw_text": "",
            "error": "Gemini API key is not configured. Set GEMINI_API_KEY in Streamlit secrets or environment variables.",
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
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        raw_text = getattr(response, "text", "") or ""
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
                "error": "Gemini returned non-JSON text. Raw response captured.",
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
            "error": f"Gemini commentary failed: {exc}",
        }