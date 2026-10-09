import json
from pathlib import Path
import tempfile
import unittest
from . import run


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[2]
        cls.cases = run.read(cls.repo / 'data/exports/eduskillbench-305-20261003/cases.json')

    def test_question_only_and_native_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); c = self.cases[0]
            for mode in ['no_skill', 'with_skill']:
                task = run.build_task(c, mode, out, self.repo / 'skills/single_turn', 'test-image', {})
                self.assertEqual((task / 'instruction.md').read_text(), c['context'] + '\n\n' + c['user_prompt'] + '\n')
                self.assertNotIn('## Required procedure', (task / 'instruction.md').read_text())
                if mode == 'with_skill':
                    self.assertEqual([p.name for p in (task / 'environment/skills').iterdir()], [c['task_id'].split('__')[0]])
                    for p in (task / 'environment/skills').rglob('*'):
                        if p.is_file(): self.assertTrue(p.stat().st_mode & 4)

    def test_candidate_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            c = self.cases[0]
            with self.assertRaises(ValueError):
                run.build_task(c, 'with_skill', Path(temp), self.repo / 'skills/single_turn', 'x', {c['task_id']: ['../secrets']})

    def test_terminal_parent_only(self):
        db = {'session': [{'id': 'root'}, {'id': 'child', 'parent_id': 'root'}],
              'message': [{'id': 'u', 'session_id': 'root', 'data': {'role': 'user'}},
                          {'id': 'a', 'session_id': 'root', 'data': {'role': 'assistant', 'finish': 'stop'}},
                          {'id': 'b', 'session_id': 'child', 'data': {'role': 'assistant', 'finish': 'stop'}}],
              'part': [{'id': 'p', 'message_id': 'a', 'data': {'type': 'text', 'text': 'final'}},
                       {'id': 'q', 'message_id': 'b', 'data': {'type': 'text', 'text': 'ignore'}}]}
        text, info = run.final_answer(db)
        self.assertEqual(text, 'final'); self.assertTrue(info['terminal'])

    def test_read_is_not_functional_adoption(self):
        c = self.cases[0]; expected = c['task_id'].split('__')[0]
        db = {'part': [{'data': {'type': 'tool', 'tool': 'skill', 'state': {'status': 'completed', 'input': {'name': expected}}}}]}
        audit = run.route_audit(db, c, [expected])
        self.assertTrue(audit['matched_skill_tool_completed'])
        self.assertIn('not implied', audit['functional_adoption'])

    def test_dataset_counts_and_identity(self):
        self.assertEqual(len(self.cases), 305)
        self.assertEqual(sum(c['suite'] == 'advisory' for c in self.cases), 263)
        self.assertEqual(run.sha(self.repo / 'data/exports/eduskillbench-305-20261003/cases.json'),
                         'ab0eeee5246679373d7ceb23bd87f80630c87698150d0fc49085ec7fc369f777')


if __name__ == '__main__': unittest.main()
