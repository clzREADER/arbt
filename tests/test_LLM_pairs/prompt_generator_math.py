import json
import os

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SAMPLES_FILE = os.path.join(_SOURCE_DIR, 'data_sin.json')
_TEMPLATES_DIR = os.path.join(_SOURCE_DIR, 'templates')
_DEFAULT_TEMPLATE_FILE = os.path.join(_TEMPLATES_DIR, 'template_sin.txt')

_samples_cache = {}
_template_cache = {}


def _load_samples(samples_file):
    path = os.path.abspath(samples_file)
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
    path = template_file
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(_TEMPLATES_DIR, template_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template file '{template_file}' not found.")
    path = os.path.abspath(path)
    if path not in _template_cache:
        with open(path, 'r', encoding='utf-8') as f:
            _template_cache[path] = f.read()
    return _template_cache[path]


def _format_params(sample, use_parsed):
    x_key = 'x' if use_parsed else 'x_text'
    params = dict(sample.get(x_key, {}))
    return "\n".join(f"- {k}: {v}" for k, v in params.items())


def generate_prompt(
    left_id: str,
    right_id: str,
    samples_file: str = _DEFAULT_SAMPLES_FILE,
    template_file: str = _DEFAULT_TEMPLATE_FILE,
    use_parsed: bool = True,
) -> str:
    """
    Генерирует промпт для LLM для сравнения двух входных вариантов функции f(x).

    Args:
        left_id -- id первого кандидата (Variant A)
        right_id -- id второго кандидата (Variant B)
        samples_file -- путь к файлу с датасетом (data_sin.json / data_sphere.json)
        template_file -- путь к файлу шаблона (template_sin.txt / template_sphere.txt)
        use_parsed -- использовать разобранные параметры x (иначе x_text)

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

    template = _load_template(template_file)

    variant_a_text = _format_params(left_sample, use_parsed)
    variant_b_text = _format_params(right_sample, use_parsed)

    prompt = template.replace('[Variant A]', variant_a_text)
    prompt = prompt.replace('[Variant B]', variant_b_text)

    return prompt
