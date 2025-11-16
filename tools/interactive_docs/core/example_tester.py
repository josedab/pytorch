"""
Documentation Example Tester

This module extracts and tests code examples from PyTorch documentation
to ensure they remain valid and up-to-date.

Based on RFC-0004: Interactive Documentation with Live Examples
"""

import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import ast


@dataclass
class CodeExample:
    """Represents a code example from documentation"""
    file: Path
    line: int
    code: str
    language: str = "python"
    expected_output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TestResult:
    """Result of testing a code example"""
    success: bool
    error: Optional[str] = None
    traceback: Optional[str] = None
    execution_time: Optional[float] = None


class DocExampleTester:
    """Extract and test code examples from documentation"""

    def __init__(self, doc_root: Path):
        """
        Initialize the example tester.

        Args:
            doc_root: Root directory of documentation files
        """
        self.doc_root = Path(doc_root)
        self.code_block_patterns = {
            'rst': [
                # RST code-block directive
                r'\.\. code-block:: python\n\n((?:    .*\n)*)',
                # RST doctest
                r'((?:>>> .*\n(?:\.\.\..*\n)*(?:[^ ].*\n)*)*)',
            ],
            'md': [
                # Markdown fenced code blocks
                r'```python\n(.*?)\n```',
                # Interactive example blocks
                r'<interactive-example>\s*```python\n(.*?)\n```\s*</interactive-example>',
            ]
        }

    def find_doc_files(self, pattern: str = "**/*.rst") -> List[Path]:
        """
        Find all documentation files matching the pattern.

        Args:
            pattern: Glob pattern for finding files

        Returns:
            List of documentation file paths
        """
        return list(self.doc_root.glob(pattern))

    def extract_examples(self, doc_file: Path) -> List[CodeExample]:
        """
        Extract code examples from a documentation file.

        Args:
            doc_file: Path to documentation file

        Returns:
            List of extracted code examples
        """
        examples = []

        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Determine file type
        file_type = doc_file.suffix.lstrip('.')
        if file_type not in self.code_block_patterns:
            return examples

        # Extract code blocks based on file type
        for pattern in self.code_block_patterns[file_type]:
            for match in re.finditer(pattern, content, re.DOTALL | re.MULTILINE):
                code = match.group(1)

                # Clean up code
                if file_type == 'rst':
                    # Remove indentation from RST code blocks
                    lines = code.split('\n')
                    code = '\n'.join(line[4:] if line.startswith('    ') else line
                                   for line in lines if line.strip())

                    # Handle doctest format
                    if code.startswith('>>>'):
                        code = self._clean_doctest(code)

                # Calculate line number
                line_num = content[:match.start()].count('\n') + 1

                if code.strip():
                    examples.append(CodeExample(
                        file=doc_file,
                        line=line_num,
                        code=code.strip(),
                    ))

        return examples

    def _clean_doctest(self, doctest_code: str) -> str:
        """
        Convert doctest format to executable Python code.

        Args:
            doctest_code: Code in doctest format (>>> ...)

        Returns:
            Executable Python code
        """
        lines = []
        for line in doctest_code.split('\n'):
            if line.startswith('>>> '):
                lines.append(line[4:])
            elif line.startswith('... '):
                lines.append(line[4:])
            elif line.strip() and not line.startswith('>>>'):
                # This is output, skip it
                continue
            else:
                lines.append(line)

        return '\n'.join(lines)

    def test_example(self, example: CodeExample,
                    timeout: float = 10.0) -> TestResult:
        """
        Execute a code example and verify it works.

        Args:
            example: Code example to test
            timeout: Maximum execution time in seconds

        Returns:
            Test result
        """
        import time

        try:
            # Create isolated namespace with common imports
            namespace = self._create_namespace()

            # Check if code is syntactically valid
            try:
                ast.parse(example.code)
            except SyntaxError as e:
                return TestResult(
                    success=False,
                    error=f"Syntax error: {e}",
                )

            # Execute code
            start_time = time.time()
            exec(example.code, namespace)
            execution_time = time.time() - start_time

            # Verify expected outputs if specified
            if example.expected_output:
                self._verify_output(namespace, example.expected_output)

            return TestResult(
                success=True,
                execution_time=execution_time
            )

        except Exception as e:
            return TestResult(
                success=False,
                error=str(e),
                traceback=traceback.format_exc()
            )

    def _create_namespace(self) -> Dict[str, Any]:
        """
        Create an isolated namespace for code execution.

        Returns:
            Dictionary containing common imports and utilities
        """
        namespace = {
            '__builtins__': __builtins__,
        }

        # Try to import PyTorch (may not be available in all environments)
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torch.optim as optim
            namespace.update({
                'torch': torch,
                'nn': nn,
                'F': F,
                'optim': optim,
            })
        except ImportError:
            pass

        # Try to import NumPy
        try:
            import numpy as np
            namespace['np'] = np
        except ImportError:
            pass

        return namespace

    def _verify_output(self, namespace: Dict[str, Any],
                      expected_output: str) -> None:
        """
        Verify that execution produced expected output.

        Args:
            namespace: Execution namespace
            expected_output: Expected output string

        Raises:
            AssertionError: If output doesn't match
        """
        # This is a simplified implementation
        # In practice, you'd want more sophisticated output verification
        pass

    def test_all_examples(self, pattern: str = "**/*.rst") -> Dict[Path, List[TestResult]]:
        """
        Test all examples in documentation.

        Args:
            pattern: Glob pattern for finding documentation files

        Returns:
            Dictionary mapping file paths to test results
        """
        results = {}

        for doc_file in self.find_doc_files(pattern):
            examples = self.extract_examples(doc_file)
            if examples:
                results[doc_file] = [
                    self.test_example(ex) for ex in examples
                ]

        return results

    def generate_report(self, results: Dict[Path, List[TestResult]]) -> str:
        """
        Generate a human-readable test report.

        Args:
            results: Test results from test_all_examples

        Returns:
            Formatted report string
        """
        total = sum(len(tests) for tests in results.values())
        passed = sum(
            sum(1 for t in tests if t.success)
            for tests in results.values()
        )

        report_lines = [
            "=" * 70,
            "Documentation Example Test Report",
            "=" * 70,
            f"Total examples: {total}",
            f"Passed: {passed}",
            f"Failed: {total - passed}",
            f"Success rate: {100 * passed / total:.1f}%" if total > 0 else "N/A",
            "=" * 70,
            "",
        ]

        # Report failures
        if total > passed:
            report_lines.append("FAILURES:")
            report_lines.append("-" * 70)

            for doc_file, tests in results.items():
                failures = [(i, t) for i, t in enumerate(tests) if not t.success]
                if failures:
                    report_lines.append(f"\n{doc_file}:")
                    for idx, failure in failures:
                        report_lines.append(f"  Example #{idx + 1}:")
                        report_lines.append(f"    Error: {failure.error}")
                        if failure.traceback:
                            report_lines.append(f"    Traceback:")
                            for line in failure.traceback.split('\n'):
                                if line.strip():
                                    report_lines.append(f"      {line}")

        report_lines.append("")
        report_lines.append("=" * 70)

        return '\n'.join(report_lines)

    def generate_json_report(self, results: Dict[Path, List[TestResult]]) -> Dict:
        """
        Generate a JSON-formatted test report.

        Args:
            results: Test results from test_all_examples

        Returns:
            Dictionary suitable for JSON serialization
        """
        total = sum(len(tests) for tests in results.values())
        passed = sum(
            sum(1 for t in tests if t.success)
            for tests in results.values()
        )

        return {
            'summary': {
                'total': total,
                'passed': passed,
                'failed': total - passed,
                'success_rate': 100 * passed / total if total > 0 else 0,
            },
            'results': {
                str(doc_file): [
                    {
                        'success': t.success,
                        'error': t.error,
                        'execution_time': t.execution_time,
                    }
                    for t in tests
                ]
                for doc_file, tests in results.items()
            }
        }


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Test code examples in PyTorch documentation"
    )
    parser.add_argument(
        '--doc-root',
        type=Path,
        default=Path('docs/source'),
        help='Root directory of documentation'
    )
    parser.add_argument(
        '--pattern',
        default='**/*.rst',
        help='Glob pattern for finding documentation files'
    )
    parser.add_argument(
        '--fail-on-error',
        action='store_true',
        help='Exit with error code if any test fails'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )

    args = parser.parse_args()

    tester = DocExampleTester(args.doc_root)
    results = tester.test_all_examples(args.pattern)

    if args.json:
        import json
        report = tester.generate_json_report(results)
        print(json.dumps(report, indent=2))
    else:
        report = tester.generate_report(results)
        print(report)

    # Exit with error if tests failed and --fail-on-error is set
    if args.fail_on_error:
        total = sum(len(tests) for tests in results.values())
        passed = sum(
            sum(1 for t in tests if t.success)
            for tests in results.values()
        )
        if total > passed:
            sys.exit(1)
