"""Validate Labelme insulator polygons and report their coverage of the review list."""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json-dir', default='obb_dataset/images/train')
    p.add_argument('--manifest', default='mask_review/train/manifest.csv')
    a = p.parse_args()
    task_stems = {Path(row['image']).stem for row in csv.DictReader(Path(a.manifest).open(encoding='utf-8-sig'))}
    labels, invalid, valid, valid_tasks = Counter(), [], 0, 0
    files = sorted(Path(a.json_dir).glob('*.json'))
    for path in files:
        try:
            shapes = json.loads(path.read_text(encoding='utf-8')).get('shapes', [])
        except (OSError, json.JSONDecodeError):
            invalid.append(path.name); continue
        labels.update(shape.get('label', '') for shape in shapes)
        ok = any(shape.get('label') == 'insulator' and shape.get('shape_type') == 'polygon' and len(shape.get('points', [])) >= 3 for shape in shapes)
        if ok:
            valid += 1
            valid_tasks += path.stem in task_stems
        else:
            invalid.append(path.name)
    print(f'Labelme JSON: {len(files)}')
    print(f'Valid insulator polygons: {valid}')
    print(f'Valid polygons in 168-image task list: {valid_tasks}/{len(task_stems)}')
    print(f'Invalid/missing insulator polygon: {len(invalid)}')
    print(f'Labels: {dict(labels)}')
    if invalid: print('Examples:', ', '.join(invalid[:10]))

if __name__ == '__main__': main()
