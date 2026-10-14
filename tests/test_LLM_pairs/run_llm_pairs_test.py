import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE))

from pairwise_llm import run_pairwise_llm  # noqa: E402
from prompt_generator_math import generate_prompt as generate_prompt_math  # noqa: E402

FUNCTIONS = {
    'sin': {
        'data': HERE / 'data_sin.json',
        'template': HERE / 'templates' / 'template_sin.txt',
        'formula': 'f(x) = sin(x)',
    },
    'sphere': {
        'data': HERE / 'data_sphere.json',
        'template': HERE / 'templates' / 'template_sphere.txt',
        'formula': 'f(x1, x2) = x1^2 + x2^2',
    },
}

CLOSE_LIMIT = 0.1


def load_split(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    y_map = {str(s['id']): s['y'] for s in samples}
    K_list = [str(s['id']) for s in samples if s.get('split') == 'K']
    L_list = [str(s['id']) for s in samples if s.get('split') == 'L']
    return samples, K_list, L_list, y_map


def expected_choice(y_left, y_right):
    # plan.txt: if y_i == y_j, treat y_i (variant A) as the greater one
    return 'A' if y_left >= y_right else 'B'


def evaluate(records, y_map):
    n = len(records)
    correct = 0
    c_vals = []
    p_vals = []
    errors = []
    for record in records:
        expected = expected_choice(y_map[record['left_id']], y_map[record['right_id']])
        is_correct = record['choice'] == expected
        correct += is_correct
        c_vals.append(record['c_ij'])
        p_vals.append(record['p_ij'])
        if not is_correct:
            errors.append({
                'left_id': record['left_id'],
                'right_id': record['right_id'],
                'choice': record['choice'],
                'expected': expected,
            })
    p_close = sum(1 for p in p_vals if abs(p - 0.5) >= 0.5 - CLOSE_LIMIT)
    metrics = {
        'n_pairs': n,
        'n_correct': correct,
        'accuracy': correct / n if n else 0.0,
        'c_min': min(c_vals) if c_vals else None,
        'c_mean': sum(c_vals) / n if n else None,
        'c_max': max(c_vals) if c_vals else None,
        'p_min': min(p_vals) if p_vals else None,
        'p_mean': sum(p_vals) / n if n else None,
        'p_max': max(p_vals) if p_vals else None,
        'p_frac_close_to_0_or_1': p_close / n if n else 0.0,
        'errors': errors,
    }
    return metrics


def print_report(name, formula, metrics):
    print(f"\n=== {name}: {formula} ===")
    print(f"pairs:            {metrics['n_pairs']}")
    print(f"correct:          {metrics['n_correct']}/{metrics['n_pairs']} "
          f"(accuracy {metrics['accuracy']:.3f})")
    print(f"c_ij  min/mean/max: {metrics['c_min']:.4f} / {metrics['c_mean']:.4f} / {metrics['c_max']:.4f}")
    print(f"p_ij  min/mean/max: {metrics['p_min']:.4f} / {metrics['p_mean']:.4f} / {metrics['p_max']:.4f}")
    print(f"p_ij close to 0 or 1 (>= {1 - CLOSE_LIMIT:.1f}): "
          f"{metrics['p_frac_close_to_0_or_1']:.3f}")
    if metrics['errors']:
        print("errors:")
        for err in metrics['errors']:
            print(f"  {err['left_id']} vs {err['right_id']}: "
                  f"got {err['choice']}, expected {err['expected']}")


def dry_run(name, cfg):
    samples, K_list, L_list, _ = load_split(cfg['data'])
    prompt_fn = partial(
        generate_prompt_math,
        samples_file=str(cfg['data']),
        template_file=str(cfg['template']),
    )
    n_pairs = len(K_list) * (len(set(K_list) | set(L_list))) - len(K_list)
    print(f"\n=== DRY RUN {name}: {cfg['formula']} ===")
    print(f"K={len(K_list)} L={len(L_list)} pairs (all_pairs=True): {n_pairs}")
    sample = samples[0]
    print(f"sample record: id={sample['id']} x={sample['x']} y={sample['y']}")
    print("--- prompt ---")
    print(prompt_fn(K_list[0], K_list[1]))


def run_function(name, cfg, max_workers):
    samples, K_list, L_list, y_map = load_split(cfg['data'])
    prompt_fn = partial(
        generate_prompt_math,
        samples_file=str(cfg['data']),
        template_file=str(cfg['template']),
    )
    records = run_pairwise_llm(
        K_list,
        L_list,
        str(cfg['data']),
        llm_params={'max_workers': max_workers},
        all_pairs=True,
        prompt_fn=prompt_fn,
        rag_file='',
    )
    metrics = evaluate(records, y_map)
    print_report(name, cfg['formula'], metrics)
    return {
        'formula': cfg['formula'],
        'data': str(cfg['data']),
        'K_list': K_list,
        'L_list': L_list,
        'records': records,
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='LLM pairwise test on known functions.')
    parser.add_argument('--functions', nargs='+', default=list(FUNCTIONS),
                        choices=list(FUNCTIONS), help='functions to test')
    parser.add_argument('--dry-run', action='store_true',
                        help='only print prompts and pair counts, no LLM calls')
    parser.add_argument('--max-workers', type=int, default=6,
                        help='parallel LLM requests (default 6)')
    args = parser.parse_args()

    if args.dry_run:
        for name in args.functions:
            dry_run(name, FUNCTIONS[name])
        return

    results = {}
    for name in args.functions:
        results[name] = run_function(name, FUNCTIONS[name], args.max_workers)

    results_dir = HERE / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = results_dir / f'results_{timestamp}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n=== SUMMARY ===")
    for name, data in results.items():
        m = data['metrics']
        print(f"{name}: accuracy {m['accuracy']:.3f} "
              f"({m['n_correct']}/{m['n_pairs']}), "
              f"c_mean {m['c_mean']:.4f}, "
              f"p_close_frac {m['p_frac_close_to_0_or_1']:.3f}")


if __name__ == '__main__':
    main()
