"""Keep removed scientific definitions out of active production sources."""

from pathlib import Path
import re
import unittest


REPO = Path(__file__).resolve().parents[1]


class FormulationCleanupTests(unittest.TestCase):
    def test_active_production_has_no_removed_population_or_reference_logic(self):
        # Scan production and generated configuration sources, not tests, result
        # artifacts, documentation, or the documentation validator's rejection
        # vocabulary. The test intentionally excludes its own source.
        sources = {REPO / name for name in ('train.py', 'train.sh', 'infer.py', 'infer.sh')}
        text_suffixes = {'.py', '.sh', '.json', '.csv', '.toml', '.yaml', '.yml'}
        for directory in ('emulator', 'configs', 'tools'):
            sources.update(path for path in (REPO / directory).rglob('*')
                           if path.is_file() and path.suffix in text_suffixes
                           and '__pycache__' not in path.parts
                           and path != REPO / 'tools' / 'audit_documentation.py')
        forbidden = re.compile(r'top5|peak5|tail_frac|checkpoint_score_refs', re.IGNORECASE)
        violations = []
        for path in sorted(sources):
            self.assertTrue(path.is_file(), str(path))
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if forbidden.search(line):
                    violations.append(f'{path.relative_to(REPO)}:{number}: {line.strip()}')
        self.assertFalse(violations, '\n'.join(violations))
        # Prevent an accidentally empty or relocated configuration tree from
        # silently reducing this to a check of the four entry points.
        self.assertTrue(any(path.is_relative_to(REPO / 'configs' / 'current') for path in sources))
        self.assertIn(REPO / 'tools' / 'generate_configs.py', sources)
        self.assertIn(REPO / 'emulator' / 'training' / 'metrics.py', sources)


if __name__ == '__main__':
    unittest.main()
