import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SOURCE_DIR)
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'temp', '01_LLM_pairs')

DEFAULT_LLM_PARAMS = {
    'url': 'https://polza.ai/api/v1',
    'model': 'qwen/qwen-plus',
    'providers': ['alibaba'],
    'api_key_file': '../API.txt',
    'temperature': 1.0,
    'max_workers': 4,
}

DEFAULT_COEFFICIENTS = {
    'alpha': 0.986,
    'beta': 0.014,
    'gamma': 0.58,
    'delta': 1.5,
    'k': 70.0,
}

_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503}
_MAX_NETWORK_RETRIES = 8
_NETWORK_BACKOFF_BASE = 2.0
_NETWORK_BACKOFF_CAP = 30.0

# "SOLUTION: A" / "SOLUTION: B" (final decision), but not the echoed
# instruction line "SOLUTION: A (or B)".
_SOLUTION_RE = re.compile(r"SOLUTION:\s*\**([AB])\b(?!\s*\(or)", re.IGNORECASE)


class ParseError(ValueError):
    """LLM answer does not contain a parseable A/B decision."""


def _read_api_key(api_key_file):
    path = api_key_file
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = os.path.join(_SOURCE_DIR, path)
        if os.path.exists(candidate):
            path = candidate
    if not os.path.exists(path):
        raise FileNotFoundError(f"API key file '{api_key_file}' not found.")
    with open(path, 'r', encoding='utf-8') as f:
        key = f.read().strip()
    if not key:
        raise ValueError(f"API key file '{path}' is empty.")
    return key


def _load_dataset_ids(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        records = data.values()
    else:
        records = data
    ids = set()
    for record in records:
        if 'id' not in record:
            raise ValueError(f"Record without 'id' in dataset '{dataset_path}'.")
        ids.add(str(record['id']))
    return ids


def _build_pairs(K_list, L_list, all_pairs):
    K_list = list(dict.fromkeys(str(i) for i in K_list))
    L_list = list(dict.fromkeys(str(i) for i in L_list))
    if not K_list:
        raise ValueError("K_list must not be empty.")
    if all_pairs:
        right_ids = list(dict.fromkeys(K_list + L_list))
    else:
        right_ids = list(dict.fromkeys(L_list))
    pairs = []
    for left_id in K_list:
        for right_id in right_ids:
            if left_id == right_id:
                continue
            pairs.append((left_id, right_id))
    return pairs


def _default_prompt_fn(dataset_path):
    def prompt_fn(left_id, right_id):
        import sys
        if _SOURCE_DIR not in sys.path:
            sys.path.insert(0, _SOURCE_DIR)
        from prompt_generator import generate_prompt
        return generate_prompt(left_id, right_id, samples_file=dataset_path)
    return prompt_fn


def _read_rag(rag_file):
    if not rag_file:
        return ''
    if not os.path.exists(rag_file):
        print(f"Warning: RAG file '{rag_file}' not found. Skipping RAG addition.")
        return ''
    with open(rag_file, 'r', encoding='utf-8') as f:
        return f.read()


def _save_results(records, output_file):
    if output_file is None:
        os.makedirs(_DEFAULT_OUTPUT_DIR, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(_DEFAULT_OUTPUT_DIR, f'llm_pairs_{timestamp}.json')
    else:
        path = os.path.abspath(output_file)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return path


def _parse_decision(content):
    matches = list(_SOLUTION_RE.finditer(content))
    if not matches:
        raise ParseError(f"No SOLUTION: A/B decision in LLM answer.")
    match = matches[-1]
    return match.group(1).upper(), match.start(1)


def _decision_logprob(tokens, letter_pos, decision):
    cum = 0
    for token in tokens:
        text = token.token or ''
        step = max(len(text), 1)
        if cum <= letter_pos < cum + step:
            return token.logprob
        cum += step
    for token in reversed(tokens):
        if (token.token or '').strip().upper() == decision:
            return token.logprob
    return tokens[-1].logprob


def compute_confidence(tokens, L_last, coefficients):
    """c_ij = alpha * exp(L_last) + beta * exp(s / m^gamma) * (m / (m + k))^delta."""
    m = len(tokens)
    if m == 0:
        raise RuntimeError("Cannot compute confidence: no output tokens.")
    s = sum(token.logprob for token in tokens)
    alpha = coefficients['alpha']
    beta = coefficients['beta']
    gamma = coefficients['gamma']
    delta = coefficients['delta']
    k = coefficients['k']
    c = (
        alpha * math.exp(L_last)
        + beta * math.exp(s / (m ** gamma)) * ((m / (m + k)) ** delta)
    )
    return min(max(c, 0.0), 1.0)


def compute_posterior(y_ij, c_ij):
    """p_ij = y_ij * c_ij + (1 - y_ij) * (1 - c_ij)."""
    return y_ij * c_ij + (1.0 - y_ij) * (1.0 - c_ij)


def _chat_with_retries(client, model, prompt, temperature, providers):
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                logprobs=True,
                extra_body={
                    "provider": {
                        "only": list(providers),
                        "allow_fallbacks": False,
                    }
                },
            )
        except (APIConnectionError, APITimeoutError) as exc:
            retryable, status = True, 'connection'
        except APIStatusError as exc:
            retryable = exc.status_code in _RETRYABLE_STATUS_CODES
            status = exc.status_code
            if not retryable:
                raise
        if attempt >= _MAX_NETWORK_RETRIES:
            raise RuntimeError(
                f"LLM request failed after {attempt} retries (last status: {status}): {exc}"
            )
        delay = min(_NETWORK_BACKOFF_BASE * (2 ** attempt), _NETWORK_BACKOFF_CAP)
        print(f"Warning: LLM request error ({status}), retry {attempt + 1} in {delay:.1f}s")
        time.sleep(delay)
        attempt += 1


def _query_pair(client, left_id, right_id, prompt, llm_params, coefficients):
    while True:
        response = _chat_with_retries(
            client,
            llm_params['model'],
            prompt,
            llm_params['temperature'],
            llm_params['providers'],
        )
        content = response.choices[0].message.content or ''
        try:
            decision, letter_pos = _parse_decision(content)
        except ParseError as exc:
            print(f"Warning: LLM parse error for ({left_id}, {right_id}): {exc}. Re-requesting.")
            continue
        logprobs = response.choices[0].logprobs
        tokens = list(getattr(logprobs, 'content', None) or [])
        if not tokens:
            print(
                f"Warning: no logprobs for ({left_id}, {right_id}). Re-requesting."
            )
            continue
        L_last = _decision_logprob(tokens, letter_pos, decision)
        c_ij = compute_confidence(tokens, L_last, coefficients)
        y_ij = 1 if decision == 'A' else 0
        p_ij = compute_posterior(y_ij, c_ij)
        return {
            'left_id': left_id,
            'right_id': right_id,
            'choice': decision,
            'y_ij': y_ij,
            'c_ij': c_ij,
            'p_ij': p_ij,
        }


def run_pairwise_llm(
    K_list,
    L_list,
    dataset_path,
    llm_params=None,
    coefficients=None,
    all_pairs=True,
    prompt_fn=None,
    rag_file='',
    output_file=None,
):
    """
    Build pairs, query the LLM and compute confidences / posterior probabilities.

    Inputs:
        K_list -- test sample (list of ids), first member of each pair (Regime A)
        L_list -- train sample / anchors (list of ids)
        dataset_path -- path to the dataset file (json list or dict of records with 'id')

    Args:
        llm_params -- dict with keys: url (default https://polza.ai/api/v1),
            model (default qwen/qwen-plus), providers (default ["alibaba"]),
            api_key_file (default ../API.txt), temperature (default 1.0),
            max_workers (default 4)
        coefficients -- dict with keys alpha, beta, gamma, delta, k
            (defaults: 0.986, 0.014, 0.58, 1.5, 70)
        all_pairs -- True: pairs K x (K u L); False: pairs K x L
        prompt_fn -- callable(left_id, right_id) -> str
            (default: generate_prompt from prompt_generator.py)
        rag_file -- path to a RAG file appended to every prompt ('' = nothing appended)
        output_file -- where to save the resulting list of dicts.
            None (default): temp/01_LLM_pairs/llm_pairs_<timestamp>.json;
            explicit path: saved exactly there (without timestamp)

    Returns:
        list of dicts: left_id, right_id, choice ('A'/'B'), y_ij (1 if A), c_ij, p_ij
    """
    params = dict(DEFAULT_LLM_PARAMS)
    if llm_params:
        unknown = set(llm_params) - set(DEFAULT_LLM_PARAMS)
        if unknown:
            raise ValueError(f"Unknown llm_params keys: {sorted(unknown)}")
        params.update(llm_params)

    coeffs = dict(DEFAULT_COEFFICIENTS)
    if coefficients:
        unknown = set(coefficients) - set(DEFAULT_COEFFICIENTS)
        if unknown:
            raise ValueError(f"Unknown coefficients keys: {sorted(unknown)}")
        coeffs.update(coefficients)

    dataset_path = os.path.abspath(dataset_path)
    known_ids = _load_dataset_ids(dataset_path)
    for sample_id in list(K_list) + list(L_list):
        if str(sample_id) not in known_ids:
            raise ValueError(f"id '{sample_id}' not found in dataset '{dataset_path}'")

    pairs = _build_pairs(K_list, L_list, all_pairs)
    if not pairs:
        raise ValueError("No pairs to compare (empty K_list or L_list?).")

    if prompt_fn is None:
        prompt_fn = _default_prompt_fn(dataset_path)

    rag_content = _read_rag(rag_file)

    prompts = []
    for left_id, right_id in pairs:
        prompt = prompt_fn(left_id, right_id)
        if rag_content:
            prompt = prompt + "\n\n" + rag_content
        prompts.append(prompt)

    api_key = _read_api_key(params['api_key_file'])
    client = OpenAI(api_key=api_key, base_url=params['url'])

    results = [None] * len(pairs)
    max_workers = max(int(params['max_workers']), 1)

    def worker(index):
        left_id, right_id = pairs[index]
        record = _query_pair(client, left_id, right_id, prompts[index], params, coeffs)
        print(f"[{index + 1}/{len(pairs)}] {left_id} vs {right_id}: {record['choice']} "
              f"c_ij={record['c_ij']:.4f} p_ij={record['p_ij']:.4f}")
        return index, record

    if max_workers == 1:
        for index in range(len(pairs)):
            idx, record = worker(index)
            results[idx] = record
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, index) for index in range(len(pairs))]
            for future in as_completed(futures):
                idx, record = future.result()
                results[idx] = record

    saved_path = _save_results(results, output_file)
    print(f"LLM pairs results saved: {saved_path}")

    return results
