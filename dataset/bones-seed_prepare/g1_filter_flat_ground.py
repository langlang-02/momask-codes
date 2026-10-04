#!/usr/bin/env python3
"""Filter existing MimicLite splits by source descriptions; never alter motions/latents."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


RULES = {
    'stairs_ladder': r'\b(?:stairs?|staircase|ladders?|scaffolding)\b',
    'elevated_support': r'\b(?:jump\w*|leap\w*|step\w*|walk\w*|get\w*|climb\w*|com\w*)\b[^.;]{0,65}\b(?:on|onto|off|from|up|down)\b[^.;]{0,45}\b(?:box|boxes|platform|ledge)\b',
    'seated_support': r'\b(?:sit\w*|sat|seat\w*|sited|lie|lies|lying|lays?|rest\w*)\b[^.;]{0,90}\b(?:chair|bench|stool|sofa|couch|bed|sunbed|box|table|armrests?)\b',
    'wall_support': r'\b(?:lean\w*|support\w*|push\w*|flip\w*|climb\w*)\b[^.;]{0,80}\b(?:wall|railing|handrail|fence)\b',
    'wall_launch': r'\b(?:jump\w*|run\w*)\b[^.;]{0,30}\b(?:on|off|from|against)\b[^.;]{0,30}\bwall\b',
    'supported_exercise': r'\b(?:dips|handstand push|legs on|feet on)\b[^.;]{0,60}\bbox\b',
    'vault_support': r'\bvault\w*\b[^.;]{0,60}\b(?:box|obstacle|object|fence)\b',
    'mobility_support': r'\bcrutch(?:es)?\b',
    'ledge_balance': r'\bbalanc\w*\b[^.;]{0,60}\b(?:ledge|beam)\b',
    'handrail_support': r'\b(?:hold\w*|hang\w*|grip\w*)\b[^.;]{0,60}\b(?:handrail|railing)\b',
    'pushing_large_support': r'\b(?:push\w*|lean\w*)\b[^.;]{0,60}\b(?:obstacle|large box|big box|large object)\b',
}
PATTERNS = {name: re.compile(pattern, re.I) for name, pattern in RULES.items()}


def classify(fields):
    evidence = []
    for field, text in fields.items():
        for reason, pattern in PATTERNS.items():
            match = pattern.search(text.replace('_', ' '))
            if match:
                evidence.append(dict(reason=reason, field=field, text=text, match=match.group()))
    # Some descriptions split the hand-support action and wall context across fields.
    support = {k:v for k,v in fields.items() if re.search(r'\bsupport\w*\b', v, re.I)}
    wall = {k:v for k,v in fields.items() if re.search(r'\bwall\b', v, re.I)}
    if support and wall:
        evidence.append(dict(reason='wall_support', field='combined_descriptions',
                             text=json.dumps({**support, **wall}), match='support + wall'))
    return evidence


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--feature-bank', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f'Refusing to replace existing audit: {a.output}')
    metadata = {}
    with a.metadata.open() as f:
        for row in csv.DictReader(f):
            source = row['move_g1_path']
            if not source:
                continue
            fields = {k: v for k, v in row.items() if v and ('desc' in k or k == 'content_name')}
            # Some rows label come_down_50cm_box files as a dance routine.
            # Keep the descriptive source filename as independent evidence.
            fields['source_motion_name'] = row['filename'].split('__')[0]
            if source in metadata:
                raise ValueError(f'Duplicate source: {source}')
            # Natural description 4 has observed mismatches (one-leg idle described
            # as seated guitar playing). A sole hit there requires review.
            primary = {k:v for k,v in fields.items() if k != 'content_natural_desc_4'}
            evidence = classify(primary)
            review = [] if evidence else classify({'content_natural_desc_4':fields.get('content_natural_desc_4','')})
            metadata[source] = dict(content_name=row['content_name'], evidence=evidence, review=review)
    windows_path = a.feature_bank / 'metadata/windows.jsonl'
    windows = {}
    with windows_path.open() as f:
        for line in f:
            row = json.loads(line)
            assert row['sample_id'] not in windows
            assert row['source'] in metadata, row['source']
            windows[row['sample_id']] = row
    kept, excluded, reviews, summary, reasons = {}, [], [], {}, Counter()
    seen = set()
    for split in ('train', 'val', 'test'):
        original = (a.dataset / f'{split}.txt').read_text().splitlines()
        kept[split] = []
        for motion in original:
            sample = Path(motion).stem
            assert sample not in seen, sample
            seen.add(sample)
            row = windows[sample]
            assert row['split'] == split
            info = metadata[row['source']]
            if info['evidence']:
                excluded.append(dict(split=split, motion=motion, sample_id=sample,
                                     source=row['source'], **info))
                reasons.update(set(e['reason'] for e in info['evidence']))
            else:
                kept[split].append(motion)
                if info['review']:
                    reviews.append(dict(split=split, motion=motion, sample_id=sample,
                                        source=row['source'], **info))
        summary[split] = dict(original=len(original), kept=len(kept[split]),
                              excluded=len(original)-len(kept[split]))
    assert seen == set(windows), 'Dataset and feature-bank sample sets differ'
    a.output.mkdir(parents=True)
    for split, names in kept.items():
        (a.output/f'{split}.txt').write_text(''.join(n+'\n' for n in names))
    with (a.output/'excluded.jsonl').open('w') as f:
        for row in excluded:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (a.output/'review.jsonl').open('w') as f:
        for row in reviews:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    report = dict(method='source-description rules; all windows of matching source excluded',
                  scope='text screening only; retained motions are not certified physically trackable',
                  dataset=str(a.dataset), metadata=str(a.metadata), windows=str(windows_path),
                  counts=summary, exclusion_reasons=dict(reasons), rules=RULES,
                  review_retained=len(reviews), review_policy='sole desc_4 hits retained pending review',
                  sha256={str(path):sha(path) for path in
                          [a.metadata, windows_path, Path(__file__),
                           *[a.dataset/f'{s}.txt' for s in kept],
                           *[a.output/f'{s}.txt' for s in kept], a.output/'excluded.jsonl', a.output/'review.jsonl']})
    (a.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
