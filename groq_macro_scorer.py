"""Groq/Llama 3.3 AI scoring utilities for MOIL Macro Radar.

The module never hardcodes, prints or displays API keys. It reads credentials
from Streamlit secrets or environment variables.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_KEY_NAME = "MOIL Macro Radar Groq Key"
DEFAULT_GROQ_PROJECT = "MOIL Macro Radar"


def _get_streamlit_secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return str(os.getenv(name, default) or "").strip()


def get_groq_config() -> Dict[str, str]:
    return {
        "api_key": _get_streamlit_secret("GROQ_API_KEY"),
        "model": _get_streamlit_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "key_name": _get_streamlit_secret("GROQ_KEY_NAME", DEFAULT_GROQ_KEY_NAME),
        "project": _get_streamlit_secret("GROQ_PROJECT", DEFAULT_GROQ_PROJECT),
    }


def is_groq_configured() -> bool:
    return bool(get_groq_config()["api_key"])


def _strip_json_fences(text: str) -> str:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?", "", clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r"```$", "", clean).strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        return clean[start : end + 1]
    return clean


def _safe_records(df: Optional[pd.DataFrame], limit: int = 50) -> list:
    if df is None or df.empty:
        return []
    out = df.copy().head(limit)
    return json.loads(out.to_json(orient="records", date_format="iso"))


def _call_groq_json(prompt: str, temperature: float = 0.1, max_tokens: int = 1800) -> Dict[str, Any]:
    config = get_groq_config()
    if not config["api_key"]:
        return {"ok": False, "error": "GROQ_API_KEY is not configured.", "raw_text": ""}
    try:
        from groq import Groq
    except Exception as exc:
        return {"ok": False, "error": f"groq package is not installed: {exc}", "raw_text": ""}

    try:
        client = Groq(api_key=config["api_key"])
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a disciplined institutional commodity analyst. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content or ""
        parsed = json.loads(_strip_json_fences(raw_text))
        parsed["ok"] = True
        parsed["raw_text"] = raw_text
        return parsed
    except TypeError:
        # Some SDK/model combinations may not support response_format. Retry without it.
        try:
            from groq import Groq

            client = Groq(api_key=config["api_key"])
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": "You are a disciplined institutional commodity analyst. Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_text = response.choices[0].message.content or ""
            parsed = json.loads(_strip_json_fences(raw_text))
            parsed["ok"] = True
            parsed["raw_text"] = raw_text
            return parsed
        except Exception as exc:
            return {"ok": False, "error": f"Groq JSON call failed: {exc}", "raw_text": ""}
    except Exception as exc:
        return {"ok": False, "error": f"Groq JSON call failed: {exc}", "raw_text": ""}


def build_macro_scoring_prompt(
    manual_macro_df: pd.DataFrame,
    market_snapshot_df: Optional[pd.DataFrame] = None,
    regime_context: Optional[dict] = None,
    news_tracker_df: Optional[pd.DataFrame] = None,
) -> str:
    context = {
        "manual_macro_inputs": _safe_records(manual_macro_df),
        "market_snapshot": _safe_records(market_snapshot_df, limit=20),
        "news_and_institutional_tracker": _safe_records(news_tracker_df, limit=30),
        "current_regime_context": regime_context or {},
        "scoring_scale": {
            "+2.0": "strongly bullish for MOIL-linked manganese/ferroalloy cycle",
            "+1.0": "bullish / constructive",
            "+0.5": "mildly constructive",
            "0.0": "neutral, mixed, stale or unavailable",
            "-0.5": "mildly negative",
            "-1.0": "bearish",
            "-1.5": "strongly bearish",
            "-2.0": "severe stress",
        },
    }
    return f"""
You are an institutional commodity analyst.

Score each macro indicator from -2 to +2 for its impact on MOIL Ltd's manganese and ferroalloy cycle.
Use only the inputs provided. Do not hallucinate missing data. If data is stale, uncertain or qualitative, reduce confidence and keep the score near zero.

Pay special attention to:
- Silico-manganese prices and demand tone
- India crude steel production / consumption
- China steel exports and dumping pressure
- China power or industrial stress
- Brent, USDINR, NIFTY Metal, MOIL trend and volume confirmation
- Institutional/news tracker where provided

Return valid JSON only with this exact schema:
{{
  "macro_scores": [
    {{
      "indicator": "string",
      "score": number between -2 and 2,
      "status": "bullish|constructive|neutral|watch|bearish|stress",
      "rationale": "short rationale using only provided facts",
      "confidence": number between 0 and 1,
      "data_quality": "high|medium|low"
    }}
  ],
  "overall_manual_macro_score": number,
  "regime_commentary": "2-4 sentence institutional read-through",
  "watch_items": ["string", "string"],
  "risks": ["string", "string"]
}}

Context JSON:
{json.dumps(context, indent=2, default=str)}
""".strip()


def _validate_macro_scores(result: Dict[str, Any], manual_macro_df: pd.DataFrame) -> Dict[str, Any]:
    if not result.get("ok"):
        return result
    macro_scores = result.get("macro_scores", [])
    if not isinstance(macro_scores, list):
        macro_scores = []
    cleaned = []
    for item in macro_scores:
        if not isinstance(item, dict):
            continue
        indicator = str(item.get("indicator", "")).strip()
        if not indicator:
            continue
        try:
            score = float(item.get("score", 0))
        except Exception:
            score = 0.0
        try:
            confidence = float(item.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        cleaned.append(
            {
                "indicator": indicator,
                "score": float(np.clip(score, -2, 2)),
                "status": str(item.get("status", "neutral")).strip() or "neutral",
                "rationale": str(item.get("rationale", "")).strip(),
                "confidence": float(np.clip(confidence, 0, 1)),
                "data_quality": str(item.get("data_quality", "medium")).strip() or "medium",
            }
        )

    # Fallback to manual rows if model returned incomplete JSON.
    manual = manual_macro_df.copy() if manual_macro_df is not None else pd.DataFrame()
    for _, row in manual.iterrows():
        indicator = str(row.get("indicator", "")).strip()
        if indicator and indicator.lower() not in {x["indicator"].lower() for x in cleaned}:
            cleaned.append(
                {
                    "indicator": indicator,
                    "score": float(np.clip(pd.to_numeric(row.get("score", 0), errors="coerce"), -2, 2)),
                    "status": str(row.get("status", "neutral")),
                    "rationale": str(row.get("commentary", "Manual fallback score.")),
                    "confidence": 0.55,
                    "data_quality": "manual fallback",
                }
            )
    result["macro_scores"] = cleaned
    if cleaned:
        result["overall_manual_macro_score"] = float(np.mean([x["score"] for x in cleaned]))
    else:
        result["overall_manual_macro_score"] = 0.0
    result.setdefault("regime_commentary", "Groq returned validated macro scores.")
    result.setdefault("watch_items", [])
    result.setdefault("risks", [])
    return result


def score_macro_with_groq(
    manual_macro_df: pd.DataFrame,
    market_snapshot_df: Optional[pd.DataFrame] = None,
    regime_context: Optional[dict] = None,
    news_tracker_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    prompt = build_macro_scoring_prompt(manual_macro_df, market_snapshot_df, regime_context, news_tracker_df)
    result = _call_groq_json(prompt, temperature=0.05, max_tokens=2200)
    if not result.get("ok"):
        return {
            "ok": False,
            "error": result.get("error", "Groq scoring failed."),
            "macro_scores": [],
            "overall_manual_macro_score": 0.0,
            "regime_commentary": "Groq unavailable; manual macro scoring remains active.",
            "watch_items": [],
            "risks": [],
            "raw_text": result.get("raw_text", ""),
        }
    return _validate_macro_scores(result, manual_macro_df)


def generate_groq_commentary(context: dict | None = None, **kwargs) -> Dict[str, Any]:
    context = context or {}
    if kwargs:
        context.update(kwargs)
    prompt = f"""
You are an institutional commodity and equity-cycle analyst writing for MOIL Macro Radar.

Generate commentary for MOIL Ltd using only the dashboard context below.
Rules:
- No direct buy/sell advice.
- Do not hallucinate missing values.
- Explain macro regime, positives, pressures, and monitoring triggers.
- Keep Telegram alert under 150 words.
- Return valid JSON only.

Return this exact JSON schema:
{{
  "telegram_alert": "under 150 words",
  "institutional_commentary": "2-4 paragraphs",
  "watchlist_triggers": ["trigger 1", "trigger 2", "trigger 3"]
}}

Dashboard context:
{json.dumps(context, indent=2, default=str)}
""".strip()
    result = _call_groq_json(prompt, temperature=0.15, max_tokens=2000)
    if not result.get("ok"):
        return {
            "ok": False,
            "telegram_alert": "",
            "institutional_commentary": "",
            "watchlist_triggers": [],
            "error": result.get("error", "Groq commentary failed."),
            "raw_text": result.get("raw_text", ""),
        }
    return {
        "ok": True,
        "telegram_alert": str(result.get("telegram_alert", "")).strip(),
        "institutional_commentary": str(result.get("institutional_commentary", "")).strip(),
        "watchlist_triggers": result.get("watchlist_triggers", []),
        "error": "",
        "raw_text": result.get("raw_text", ""),
    }


def extract_news_updates(raw_text: str) -> list:
    """Ask Groq to convert raw news into structured tracker rows."""
    prompt = f"""
Extract MOIL-relevant news, institutional activity, fund actions, downgrades, upgrades, commodity events or risk items from the text below.
Return valid JSON only:
{{
  "updates": [
    {{
      "date": "YYYY-MM-DD or blank",
      "category": "News|Institutional|Commodity|Macro|Risk",
      "item": "short item name",
      "value": "sentiment or value",
      "impact": number between -2 and 2,
      "confidence": number between 0 and 1,
      "details": "short explanation",
      "source": "source/publisher if available"
    }}
  ]
}}

Text:
{raw_text[:12000]}
""".strip()
    result = _call_groq_json(prompt, temperature=0.1, max_tokens=1800)
    if not result.get("ok"):
        return []
    updates = result.get("updates", [])
    if not isinstance(updates, list):
        return []
    cleaned = []
    for item in updates:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                "date": str(item.get("date", "")),
                "category": str(item.get("category", "News")),
                "item": str(item.get("item", ""))[:160],
                "value": str(item.get("value", ""))[:120],
                "impact": float(np.clip(pd.to_numeric(item.get("impact", 0), errors="coerce"), -2, 2)),
                "confidence": float(np.clip(pd.to_numeric(item.get("confidence", 0.5), errors="coerce"), 0, 1)),
                "details": str(item.get("details", ""))[:500],
                "source": str(item.get("source", "Groq extraction"))[:120],
            }
        )
    return cleaned
