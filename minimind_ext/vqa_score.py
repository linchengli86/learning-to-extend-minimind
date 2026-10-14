"""VQAv2 scoring with the pinned official normalization tables.

Adapted to load packaged tables instead of private experiment manifests.
See licenses/VQA.txt and third_party/vqa/ for the reference and its revision.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re


@lru_cache(maxsize=1)
def tables():
    return json.loads(Path(__file__).with_name('vqa_tables.json').read_text(encoding='utf-8'))


def punctuation(text):
    result = text
    numeric_comma = re.search(r'(\d)(\,)(\d)', text) is not None
    for mark in tables()['punct']:
        replacement = '' if (mark + ' ' in text or ' ' + mark in text or numeric_comma) else ' '
        result = result.replace(mark, replacement)
    # Preserve upstream regex and third positional argument (count=32).
    return re.compile(r'(?!<=\d)(\.)(?!\d)').sub('', result, re.UNICODE)


def digit_article(text):
    lookup = tables()
    words = [lookup['manualMap'].get(word, word) for word in text.lower().split()]
    return ' '.join(lookup['contractions'].get(word, word)
                    for word in words if word not in lookup['articles'])


def clean(text):
    return text.replace('\n', ' ').replace('\t', ' ').strip()


def score_answer(prediction, answers):
    if not isinstance(prediction, str) or not isinstance(answers, list) or len(answers) != 10:
        raise ValueError('require a string prediction and ten reference records')
    ids = [a.get('answer_id') for a in answers]
    if set(ids) != set(range(1, 11)) or any(type(i) is not int for i in ids):
        raise ValueError('reference answer_id must be exactly 1..10')
    if any(not isinstance(a.get('answer'), str) for a in answers):
        raise ValueError('invalid reference string')
    refs = [clean(a['answer']) for a in answers]
    prediction = clean(prediction)
    if len(set(refs)) > 1:
        refs = [digit_article(punctuation(a)) for a in refs]
        prediction = digit_article(punctuation(prediction))
    # IDs are validated unique, so the count formula equals upstream's dictionary
    # inequality leave-one-out loop without mutating the original annotations.
    matches = sum(a == prediction for a in refs)
    # Keep upstream's integer saturation value and argument order: Python 3.12's
    # sum can round a mixed int/float sequence differently from all floats.
    return sum(min(1, float(matches - (a == prediction)) / 3) for a in refs) / 10.0


def index(rows, kind):
    result = {}
    for row in rows:
        qid = row.get('question_id')
        if type(qid) is not int or qid < 0 or qid in result:
            raise ValueError(f'invalid/duplicate {kind} question_id')
        result[qid] = row
    return result


def score_predictions(annotations, predictions, allow_missing=False):
    refs, outputs = index(annotations, 'reference'), index(predictions, 'prediction')
    if not refs:
        raise ValueError('empty evaluation set')
    missing = set(refs) - set(outputs)
    extra = set(outputs) - set(refs)
    if extra or (missing and not allow_missing):
        raise ValueError(f'question_id mismatch: missing={len(missing)}, extra={len(extra)}')
    per_question = []
    groups = {'answer_type': defaultdict(list), 'question_type': defaultdict(list)}
    for qid, row in refs.items():
        answer = outputs[qid]['answer'] if qid in outputs else ''
        value = score_answer(answer, row['answers'])
        item = {'question_id': qid, 'image_id': row['image_id'], 'answer': answer,
                'score': value, 'answer_type': row['answer_type'], 'question_type': row['question_type']}
        per_question.append(item)
        for kind in groups:
            groups[kind][row[kind]].append(value)
    total = sum(row['score'] for row in per_question) / len(per_question)
    return {'questions': len(refs), 'missing_predictions_scored_empty': len(missing),
            'accuracy': total, 'accuracy_percent': 100 * total,
            'official_rounded_percent': round(100 * total, 2),
            'groups': {kind: {key: {'questions': len(values), 'accuracy': sum(values) / len(values),
                                   'official_rounded_percent': round(100 * sum(values) / len(values), 2)}
                              for key, values in sorted(group.items())} for kind, group in groups.items()},
            'per_question': per_question}


def main():
    parser = argparse.ArgumentParser(description='Score VQAv2 predictions on CPU')
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding='utf-8'))
    if isinstance(annotations, dict):
        annotations = annotations['annotations']
    predictions = json.loads(args.predictions.read_text(encoding='utf-8'))
    result = score_predictions(annotations, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"VQA {result['accuracy_percent']:.4f}% / {result['questions']} questions")


if __name__ == '__main__':
    main()
