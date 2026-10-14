"""Compare the portable scorer with the archived official implementation."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import unittest

from minimind_ext.vqa_score import score_answer, score_predictions

REFERENCE = Path(__file__).resolve().parents[1] / 'third_party/vqa/vqaEval_python3.py'
spec = importlib.util.spec_from_file_location('official_vqa', REFERENCE)
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


def answers(values):
    return [{'answer_id': index + 1, 'answer': value} for index, value in enumerate(values)]


class DatasetStub:
    def __init__(self, rows):
        self.qa = {row['question_id']: row for row in rows}

    def getQuesIds(self):
        return list(self.qa)


class VQAScoringTests(unittest.TestCase):
    def test_agrees_with_official_evaluator(self):
        fixtures = [
            ('two', ['2'] * 4 + ['three'] * 6),
            ('The cat!', ['cat'] * 6 + ['a dog'] * 4),
            ("dont", ["don't"] * 3 + ['yes'] * 7),
            ('1,000', ['1000'] * 5 + ['500'] * 5),
            ('3.14', ['3.14'] * 2 + ['3'] * 8),
            ('yes', ['yes'] * 10),
            ('a cat', ['cat'] * 10),
            ('red\n', ['red'] + ['blue'] * 9),
            ('', ['no'] * 10),
        ]
        references, predictions = [], []
        for qid, (prediction, values) in enumerate(fixtures):
            references.append({'question_id': qid, 'image_id': qid,
                               'answer_type': 'other', 'question_type': 'what',
                               'answers': answers(values)})
            predictions.append({'question_id': qid, 'answer': prediction})
        expected = score_predictions(references, predictions)
        evaluator = official.VQAEval(DatasetStub(references), DatasetStub(predictions))
        with redirect_stdout(io.StringIO()):
            evaluator.evaluate()
        for row in expected['per_question']:
            self.assertEqual(round(100 * row['score'], 2), evaluator.evalQA[row['question_id']])
        self.assertEqual(expected['official_rounded_percent'], evaluator.accuracy['overall'])

    def test_consensus_credit(self):
        for matches, expected in ((0, 0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1)):
            self.assertAlmostEqual(score_answer('yes', answers(['yes'] * matches + ['no'] * (10-matches))), expected)

    def test_rejects_bad_reference_ids(self):
        refs = answers(['yes'] * 10)
        refs[-1]['answer_id'] = 1
        with self.assertRaises(ValueError):
            score_answer('yes', refs)

    def test_rejects_missing_or_duplicate_predictions(self):
        refs = [{'question_id': 1, 'image_id': 1, 'answers': answers(['yes'] * 10),
                 'answer_type': 'yes/no', 'question_type': 'is'}]
        for predictions in ([], [{'question_id': 1, 'answer': 'yes'}] * 2):
            with self.assertRaises(ValueError):
                score_predictions(refs, predictions)


if __name__ == '__main__':
    unittest.main()
