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


def generate_gemini_commentary_wrapper(context: Dict[str, Any]) -> str:
    """Wrapper for Gemini commentary with fallback to rule-based."""
    commentary = generate_gemini_commentary(context)

    if commentary:
        return commentary
    else:
        # Fallback to rule-based commentary
        from macro_radar import generate_rule_based_commentary
        return "Gemini commentary unavailable. Using rule-based analysis.\n\n" + generate_rule_based_commentary(
            context.get("summary"),
            context.get("scorecard", pd.DataFrame()),
            context.get("moil_corr", pd.DataFrame()),
            context.get("anomalies", pd.DataFrame())
        )