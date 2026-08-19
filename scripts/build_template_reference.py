import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib

from app.template_reference import (
    DEFAULT_EXCLUDED_CODES,
    cargar_plantillas_desde_db,
    construir_referencia_plantillas,
)


def parse_codes(value: str) -> set[str]:
    return {item.strip() for item in value.split(',') if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description='Construye referencia estructurada desde ap_plantilla.')
    parser.add_argument('--output', default='models/auditai_template_reference.joblib')
    parser.add_argument('--exclude-codes', default=','.join(sorted(DEFAULT_EXCLUDED_CODES)))
    parser.add_argument('--max-features', type=int, default=100000)
    args = parser.parse_args()

    excluded = parse_codes(args.exclude_codes)
    templates = cargar_plantillas_desde_db(excluded)
    artifact = construir_referencia_plantillas(templates, args.max_features)
    artifact['trained_at'] = datetime.now().isoformat(timespec='seconds')
    artifact['excluded_codes'] = sorted(excluded)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    unique_codes = sorted({code for template in artifact['templates'] for code in template.get('codigos', [])})
    print(json.dumps({
        'output': str(output),
        'artifact': artifact['artifact'],
        'templates': len(artifact['templates']),
        'unique_codes': len(unique_codes),
        'excluded_codes': artifact['excluded_codes'],
        'sample_templates': artifact['templates'][:5],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
