import os
import logging
import json
import re

from app.config import load_config

logger = logging.getLogger("nexusai.azure")

_azure_available = False
_azure_client = None


def _check_azure_env():
    config = load_config()
    azure_config = config.get("azure", {})
    flag = os.environ.get(azure_config.get("env_flag", "USE_AZURE_OPENAI"), "").lower()
    if flag not in ("1", "true", "yes"):
        return False, {}

    endpoint = os.environ.get(azure_config.get("endpoint_var", "AZURE_OPENAI_ENDPOINT"), "")
    key = os.environ.get(azure_config.get("key_var", "AZURE_OPENAI_KEY"), "")
    deployment = os.environ.get(azure_config.get("deployment_var", "AZURE_OPENAI_DEPLOYMENT"), "")

    if not all([endpoint, key, deployment]):
        logger.warning("Azure flags set but missing endpoint/key/deployment — falling back to offline mode")
        return False, {}

    return True, {
        "endpoint": endpoint,
        "key": key,
        "deployment": deployment,
        "timeout": azure_config.get("timeout_seconds", 4),
    }


def is_azure_enabled():
    enabled, _ = _check_azure_env()
    return enabled


def rephrase_answer(original_answer, source_text):
    enabled, azure_params = _check_azure_env()
    if not enabled:
        return original_answer, False

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=azure_params["endpoint"],
            api_key=azure_params["key"],
            api_version="2024-02-01",
        )

        system_prompt = (
            "You are a friendly campus assistant. Rephrase the following FAQ answer "
            "in a warm, conversational tone. You MUST NOT add any facts, numbers, URLs, "
            "dates, or information that is not in the original answer. Keep all specific "
            "numbers, URLs, and dates exactly as they appear in the source. "
            "If you cannot rephrase without changing facts, return the original text unchanged."
        )

        response = client.chat.completions.create(
            model=azure_params["deployment"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original answer:\n{original_answer}"},
            ],
            max_tokens=500,
            temperature=0.3,
            timeout=azure_params["timeout"],
        )

        rephrased = response.choices[0].message.content.strip()

        if not _validate_rephrase(original_answer, rephrased):
            logger.warning("Azure rephrase validation failed — using original answer")
            return original_answer, True

        return rephrased, True

    except ImportError:
        logger.info("openai package not installed — falling back to offline mode")
        return original_answer, False
    except Exception as e:
        logger.warning(f"Azure rephrase failed: {e} — falling back to offline mode")
        return original_answer, False


def azure_second_opinion(query, offline_scores):
    enabled, azure_params = _check_azure_env()
    if not enabled:
        return None

    try:
        from openai import AzureOpenAI

        domains = list(offline_scores.keys())
        client = AzureOpenAI(
            azure_endpoint=azure_params["endpoint"],
            api_key=azure_params["key"],
            api_version="2024-02-01",
        )

        system_prompt = (
            f"You are a query router. Given a user query, classify it into one of these domains: "
            f"{', '.join(domains)}. Respond with ONLY the domain name, nothing else."
        )

        response = client.chat.completions.create(
            model=azure_params["deployment"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=20,
            temperature=0,
            timeout=azure_params["timeout"],
        )

        predicted = response.choices[0].message.content.strip().lower()
        if predicted in domains:
            return predicted
        return None

    except Exception as e:
        logger.warning(f"Azure second opinion failed: {e}")
        return None


def _validate_rephrase(original, rephrased):
    url_pattern = re.compile(r"https?://\S+|[\w.]+\.university\.edu\S*")
    original_urls = set(url_pattern.findall(original))
    rephrased_urls = set(url_pattern.findall(rephrased))

    if original_urls and not original_urls.issubset(rephrased_urls):
        return False

    number_pattern = re.compile(r"\b\d+[\d,.]*\b")
    original_numbers = set(number_pattern.findall(original))
    rephrased_numbers = set(number_pattern.findall(rephrased))

    critical_numbers = {n for n in original_numbers if len(n) >= 2}
    if critical_numbers and not critical_numbers.issubset(rephrased_numbers):
        return False

    return True
