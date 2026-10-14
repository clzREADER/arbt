import json
import os

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SOURCE_DIR)
_DEFAULT_SAMPLES_FILE = os.path.join(_PROJECT_ROOT, 'data', 'samples_rstu.json')
_TEMPLATES_DIR = os.path.join(_SOURCE_DIR, 'templates', 'prompt_rstu_data')

_samples_cache = {}
_template_cache = {}


def _resolve_samples_file(samples_file):
    if os.path.exists(samples_file):
        return samples_file
    candidate = os.path.join(_PROJECT_ROOT, 'data', samples_file)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Samples file '{samples_file}' not found.")


def _load_samples(samples_file):
    path = os.path.abspath(_resolve_samples_file(samples_file))
    if path not in _samples_cache:
        with open(path, 'r', encoding='utf-8') as f:
            samples_list = json.load(f)
        if isinstance(samples_list, dict):
            samples_dict = {str(k): v for k, v in samples_list.items()}
        else:
            samples_dict = {str(sample['id']): sample for sample in samples_list}
        _samples_cache[path] = samples_dict
    return _samples_cache[path]


def _load_template(template_file):
    path = os.path.join(_TEMPLATES_DIR, template_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template file '{path}' not found.")
    if path not in _template_cache:
        with open(path, 'r', encoding='utf-8') as f:
            _template_cache[path] = f.read()
    return _template_cache[path]


def _format_params(sample, use_parsed, use_ambiguous):
    x_key = 'x' if use_parsed else 'x_text'
    params = dict(sample.get(x_key, {}))
    if use_ambiguous:
        amb_key = 'x_ambiguous' if use_parsed else 'x_ambiguous_text'
        params.update(sample.get(amb_key, {}))
    return "\n".join(f"- {k}: {v}" for k, v in params.items())


def generate_prompt(
    left_id: str,
    right_id: str,
    samples_file: str = _DEFAULT_SAMPLES_FILE,
    use_parsed: bool = True,
    use_ambiguous: bool = True,
) -> str:
    """
    Генерирует промпт для LLM для сравнения двух режимов нанесения покрытия.

    Args:
        left_id -- id первого кандидата (Regime A)
        right_id -- id второго кандидата (Regime B)
        samples_file -- путь к файлу с датасетом (по умолчанию data/samples_rstu.json)
        use_parsed -- использовать разобранные параметры x/x_ambiguous
        use_ambiguous -- добавлять неоднозначные параметры

    Returns:
        Текст промпта (без RAG: RAG добавляется вызывающей функцией).
    """
    samples_dict = _load_samples(samples_file)

    if str(left_id) not in samples_dict:
        raise ValueError(f"Sample with id '{left_id}' not found in {samples_file}")
    if str(right_id) not in samples_dict:
        raise ValueError(f"Sample with id '{right_id}' not found in {samples_file}")

    left_sample = samples_dict[str(left_id)]
    right_sample = samples_dict[str(right_id)]

    left_ds = str(left_sample['dataset'])
    right_ds = str(right_sample['dataset'])

    if left_ds == '2022' and right_ds == '2022':
        template_file = 'exp_setup_2022.txt'
    elif left_ds == '2025' and right_ds == '2025':
        template_file = 'exp_setup_2025.txt'
    elif left_ds == '2022' and right_ds == '2025':
        template_file = 'exp_setup_2022vs2025.txt'
    elif left_ds == '2025' and right_ds == '2022':
        template_file = 'exp_setup_2025vs2022.txt'
    else:
        raise ValueError(f"Unsupported dataset combination: {left_ds} and {right_ds}")

    template = _load_template(template_file)

    regime_a_text = _format_params(left_sample, use_parsed, use_ambiguous)
    regime_b_text = _format_params(right_sample, use_parsed, use_ambiguous)

    prompt = template.replace('[Regime A]', regime_a_text)
    prompt = prompt.replace('[Regime B]', regime_b_text)

    return prompt
