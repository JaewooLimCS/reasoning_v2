"""
models/gpt.py — GPT-5 API Wrapper
===================================
단일 호출(call)과 다중 샘플링(call_n) 지원.
CoT-SC의 temperature 샘플링도 GPT-5에서 처리.

설정:
  OPENAI_API_KEY 환경변수 필요
  MODEL_NAME을 실제 GPT-5 모델명으로 수정
"""

import os
import time
from openai import OpenAI

MODEL_NAME     = "gpt-5"
TEMPERATURE    = 1            # gpt-5는 temperature=1만 지원
TEMPERATURE_SC = 1            # gpt-5는 temperature=1만 지원
MAX_COMPLETION_TOKENS = 2048
MAX_RETRIES    = 3
RETRY_DELAY    = 5

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        _client = OpenAI(api_key=api_key)
    return _client


def call(prompt: str, temperature: float = TEMPERATURE) -> str:
    """
    단일 API 호출.
    standard_io, zero_shot_cot, least_to_most, self_refine, self_discover에 사용.
    """
    client = _get_client()
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=MAX_COMPLETION_TOKENS
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [GPT] 오류 (시도 {attempt+1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise RuntimeError(f"GPT API 호출 실패 ({MAX_RETRIES}회): {e}") from e
    return ""


def call_n(prompt: str, n: int, temperature: float = TEMPERATURE_SC) -> list:
    """N개 독립 샘플링 — CoT-SC 전용."""
    return [call(prompt, temperature=temperature) for _ in range(n)]
