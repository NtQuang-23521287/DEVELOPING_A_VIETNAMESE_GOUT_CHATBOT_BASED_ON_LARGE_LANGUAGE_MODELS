"""Kiểm tra rủi ro sai dữ liệu, liên kết nhóm và rò rỉ đáp án."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import unit01_data as unit


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.rows = unit.parse_jsonl(unit.read_text(unit.DEFAULT_INPUT))

    def load_changed(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'inputs.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            return unit.load_dataset(path)

    def test_current_data_and_selected_pair(self):
        cases = unit.load_dataset(unit.DEFAULT_INPUT)
        self.assertEqual(unit.summarize(cases),
                         dict(cases=116, groups=58, turns=232, single_cases=58, multi_cases=58))
        chosen = unit.select_group(unit.group_cases(cases), 'GOUT_ST_001')
        self.assertEqual(unit.summarize(chosen)['turns'], 4)
        actual = unit.flatten_turns(chosen)
        original = [t['user'] for r in self.rows if r['group_id'] == 'GOUT_ST_001' for t in r['turns']]
        self.assertEqual([r['user'] for r in actual], original)
        self.assertTrue(all(set(r) == {'case_id','group_id','scenario','category','sample_id','turn_id','user'} for r in actual))

    def test_reference_field_at_nested_level_is_rejected(self):
        changed = copy.deepcopy(self.rows)
        changed[0]['turns'][0]['ground_truth'] = 'SECRET_REFERENCE'
        with self.assertRaisesRegex(ValueError, 'ground_truth'):
            self.load_changed(changed)

    def test_duplicate_case_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Trung case_id'):
            self.load_changed(self.rows + [self.rows[0]])

    def test_missing_multi_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, 'dung 1 single va 1 multi'):
            self.load_changed([r for r in self.rows if r['case_id'] != 'GOUT_MT_001'])

    def test_mismatched_first_question_rejected(self):
        changed = copy.deepcopy(self.rows)
        next(r for r in changed if r['case_id'] == 'GOUT_MT_001')['turns'][0]['user'] = 'Khac'
        with self.assertRaisesRegex(ValueError, 'khong khop'):
            self.load_changed(changed)

    def test_bad_turn_order_rejected(self):
        changed = copy.deepcopy(self.rows)
        next(r for r in changed if r['scenario'] == 'multi')['turns'][1]['turn_id'] = 4
        with self.assertRaisesRegex(ValueError, 'turn_id'):
            self.load_changed(changed)

    def test_json_error_has_line_number(self):
        with self.assertRaisesRegex(ValueError, 'Dong 3'):
            unit.parse_jsonl('{}\n\nnot-json')

    def test_output_does_not_overwrite_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'result'
            unit.save_output(target, [{'user': 'original'}], {})
            before = (target / 'questions.jsonl').read_bytes()
            with self.assertRaises(FileExistsError):
                unit.save_output(target, [], {})
            self.assertEqual((target / 'questions.jsonl').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
