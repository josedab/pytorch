"""
Unit tests for DocExampleTester

Tests the documentation example extraction and testing functionality.
"""

import unittest
from pathlib import Path
import tempfile
from torch.testing._internal.common_utils import TestCase, run_tests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.example_tester import DocExampleTester, CodeExample, TestResult


class TestDocExampleTester(TestCase):
    """Test cases for DocExampleTester"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.doc_root = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_rst_code_blocks(self):
        """Test extraction of code blocks from RST files"""
        # Create a test RST file
        rst_content = """
Some documentation text.

.. code-block:: python

    import torch
    x = torch.tensor([1, 2, 3])
    print(x)

More text.

.. code-block:: python

    y = torch.randn(5)
    print(y)
"""
        rst_file = self.doc_root / 'test.rst'
        rst_file.write_text(rst_content)

        # Extract examples
        tester = DocExampleTester(self.doc_root)
        examples = tester.extract_examples(rst_file)

        # Verify extraction
        self.assertEqual(len(examples), 2)
        self.assertIn('import torch', examples[0].code)
        self.assertIn('torch.randn', examples[1].code)

    def test_extract_doctest_examples(self):
        """Test extraction of doctest-style examples"""
        rst_content = """
Example usage:

>>> import torch
>>> x = torch.tensor([1, 2, 3])
>>> x
tensor([1, 2, 3])
"""
        rst_file = self.doc_root / 'test_doctest.rst'
        rst_file.write_text(rst_content)

        tester = DocExampleTester(self.doc_root)
        examples = tester.extract_examples(rst_file)

        self.assertGreater(len(examples), 0)
        # Check that doctest format was cleaned
        self.assertNotIn('>>>', examples[0].code)

    def test_extract_markdown_examples(self):
        """Test extraction from Markdown files"""
        md_content = """
# Example

```python
import torch
x = torch.tensor([1, 2, 3])
```

Another example:

```python
y = torch.randn(5)
```
"""
        md_file = self.doc_root / 'test.md'
        md_file.write_text(md_content)

        tester = DocExampleTester(self.doc_root)
        examples = tester.extract_examples(md_file)

        self.assertEqual(len(examples), 2)

    def test_execute_valid_example(self):
        """Test execution of valid code example"""
        tester = DocExampleTester(self.doc_root)

        example = CodeExample(
            file=Path('test.rst'),
            line=1,
            code='x = 1 + 1\nprint(x)'
        )

        result = tester.test_example(example)

        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_execute_invalid_example(self):
        """Test execution of code with syntax error"""
        tester = DocExampleTester(self.doc_root)

        example = CodeExample(
            file=Path('test.rst'),
            line=1,
            code='if True\n    print("missing colon")'
        )

        result = tester.test_example(example)

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_execute_runtime_error(self):
        """Test execution of code with runtime error"""
        tester = DocExampleTester(self.doc_root)

        example = CodeExample(
            file=Path('test.rst'),
            line=1,
            code='x = 1 / 0'
        )

        result = tester.test_example(example)

        self.assertFalse(result.success)
        self.assertIn('ZeroDivisionError', result.error or '')

    def test_generate_report(self):
        """Test report generation"""
        tester = DocExampleTester(self.doc_root)

        results = {
            Path('test1.rst'): [
                TestResult(success=True),
                TestResult(success=True),
            ],
            Path('test2.rst'): [
                TestResult(success=True),
                TestResult(success=False, error='Test error'),
            ],
        }

        report = tester.generate_report(results)

        # Check report contains key information
        self.assertIn('Total examples: 4', report)
        self.assertIn('Passed: 3', report)
        self.assertIn('Failed: 1', report)
        self.assertIn('test2.rst', report)
        self.assertIn('Test error', report)

    def test_generate_json_report(self):
        """Test JSON report generation"""
        tester = DocExampleTester(self.doc_root)

        results = {
            Path('test.rst'): [
                TestResult(success=True, execution_time=0.1),
                TestResult(success=False, error='Test error'),
            ],
        }

        report = tester.generate_json_report(results)

        # Check report structure
        self.assertIn('summary', report)
        self.assertIn('results', report)
        self.assertEqual(report['summary']['total'], 2)
        self.assertEqual(report['summary']['passed'], 1)
        self.assertEqual(report['summary']['failed'], 1)

    def test_find_doc_files(self):
        """Test finding documentation files"""
        # Create some test files
        (self.doc_root / 'test1.rst').write_text('content')
        (self.doc_root / 'test2.rst').write_text('content')
        (self.doc_root / 'subdir').mkdir()
        (self.doc_root / 'subdir' / 'test3.rst').write_text('content')

        tester = DocExampleTester(self.doc_root)
        files = tester.find_doc_files('**/*.rst')

        self.assertEqual(len(files), 3)

    def test_namespace_isolation(self):
        """Test that examples run in isolated namespace"""
        tester = DocExampleTester(self.doc_root)

        # First example defines a variable
        example1 = CodeExample(
            file=Path('test.rst'),
            line=1,
            code='test_var = 42'
        )

        # Second example should not see it
        example2 = CodeExample(
            file=Path('test.rst'),
            line=5,
            code='print(test_var)'
        )

        result1 = tester.test_example(example1)
        result2 = tester.test_example(example2)

        self.assertTrue(result1.success)
        self.assertFalse(result2.success)  # Should fail - test_var not defined


if __name__ == "__main__":
    run_tests()
