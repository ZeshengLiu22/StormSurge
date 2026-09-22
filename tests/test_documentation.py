"""Active method documentation must resolve against the current public interfaces."""
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.audit_documentation import ROOT, audit


class DocumentationTests(unittest.TestCase):
    def test_current_documentation_contract(self):
        result = audit()
        self.assertTrue(result['passed'], '\n'.join(result['errors']))

    def test_audit_rejects_unknown_interfaces_and_missing_metric_keys(self):
        original = Path.read_text

        def corrupted(path, *args, **kwargs):
            content = original(path, *args, **kwargs)
            if path == ROOT / 'docs' / 'METRICS.md':
                return content.replace('`all_rmse`', '`nonexistent_metric`')
            if path == ROOT / 'README.md':
                return content + '\n`--nonexistent_option` and `NONEXISTENT_CONFIG`.\n'
            return content

        with patch.object(Path, 'read_text', corrupted):
            result = audit()
        self.assertFalse(result['passed'])
        for required in ('unknown CLI option', 'unknown config variable', 'missing canonical key/label all_rmse'):
            self.assertTrue(any(required in message for message in result['errors']), result['errors'])
