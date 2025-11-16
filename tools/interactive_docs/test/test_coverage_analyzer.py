"""
Unit tests for APICoverageAnalyzer

Tests the API coverage analysis functionality.
"""

import unittest
from pathlib import Path
import tempfile
from torch.testing._internal.common_utils import TestCase, run_tests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.coverage_analyzer import APICoverageAnalyzer, DocStatus, ModuleCoverage


class TestAPICoverageAnalyzer(TestCase):
    """Test cases for APICoverageAnalyzer"""

    def test_discover_public_apis(self):
        """Test discovery of public APIs"""
        analyzer = APICoverageAnalyzer()

        # Discover torch APIs
        apis = analyzer.discover_public_apis('torch')

        # Should find common APIs
        self.assertIn('torch.tensor', apis)
        self.assertIn('torch.randn', apis)

        # Should not include private APIs
        private_apis = [api for api in apis if api.startswith('torch._')]
        self.assertEqual(len(private_apis), 0)

    def test_check_documentation_with_docstring(self):
        """Test checking documentation for API with docstring"""
        analyzer = APICoverageAnalyzer()

        # torch.tensor should have documentation
        status = analyzer.check_documentation('torch.tensor')

        self.assertTrue(status.has_docstring)
        self.assertGreater(status.docstring_length, 50)

    def test_check_documentation_type_hints(self):
        """Test checking for type hints"""
        analyzer = APICoverageAnalyzer()

        # Check various APIs for type hints
        # Note: This depends on PyTorch implementation
        status = analyzer.check_documentation('torch.tensor')

        # Just verify the check runs without error
        self.assertIsInstance(status.has_type_hints, bool)

    def test_module_coverage_percentage(self):
        """Test module coverage percentage calculation"""
        coverage = ModuleCoverage(
            total=100,
            documented=80,
            has_examples=50,
            has_type_hints=30,
        )

        self.assertEqual(coverage.doc_percentage, 80.0)
        self.assertEqual(coverage.example_percentage, 50.0)

    def test_module_coverage_zero_total(self):
        """Test module coverage with zero total APIs"""
        coverage = ModuleCoverage(
            total=0,
            documented=0,
            has_examples=0,
            has_type_hints=0,
        )

        self.assertEqual(coverage.doc_percentage, 0.0)
        self.assertEqual(coverage.example_percentage, 0.0)

    def test_analyze_coverage(self):
        """Test full coverage analysis"""
        analyzer = APICoverageAnalyzer()

        # Analyze a small subset
        report = analyzer.analyze_coverage('torch')

        # Verify report structure
        self.assertGreater(report.total_apis, 0)
        self.assertGreater(report.documented, 0)
        self.assertIsInstance(report.coverage_by_module, dict)

    def test_group_by_module(self):
        """Test grouping coverage by module"""
        analyzer = APICoverageAnalyzer()

        coverage = {
            'torch.tensor': DocStatus(
                has_docstring=True,
                has_examples=True,
                has_type_hints=False,
                has_web_docs=False,
            ),
            'torch.randn': DocStatus(
                has_docstring=True,
                has_examples=False,
                has_type_hints=False,
                has_web_docs=False,
            ),
            'torch.nn.Linear': DocStatus(
                has_docstring=True,
                has_examples=True,
                has_type_hints=True,
                has_web_docs=False,
            ),
        }

        grouped = analyzer._group_by_module(coverage)

        # Should have torch and torch.nn
        self.assertIn('torch', grouped)
        self.assertIn('torch.nn', grouped)

        # Check torch module stats
        torch_stats = grouped['torch']
        self.assertEqual(torch_stats.total, 2)
        self.assertEqual(torch_stats.documented, 2)
        self.assertEqual(torch_stats.has_examples, 1)

    def test_generate_report(self):
        """Test report generation"""
        from core.coverage_analyzer import CoverageReport

        analyzer = APICoverageAnalyzer()

        report = CoverageReport(
            total_apis=100,
            documented=80,
            has_examples=50,
            has_type_hints=30,
            coverage_by_module={
                'torch': ModuleCoverage(50, 40, 25, 15),
                'torch.nn': ModuleCoverage(30, 25, 15, 10),
            }
        )

        report_text = analyzer.generate_report(report)

        # Check report contains key information
        self.assertIn('80 / 100', report_text)
        self.assertIn('torch', report_text)
        self.assertIn('torch.nn', report_text)

    def test_import_api(self):
        """Test importing API by name"""
        analyzer = APICoverageAnalyzer()

        # Test importing valid API
        obj = analyzer._import_api('torch.tensor')
        self.assertIsNotNone(obj)

        # Test importing invalid API
        obj = analyzer._import_api('torch.nonexistent_function')
        self.assertIsNone(obj)

    def test_has_type_hints(self):
        """Test type hint detection"""
        analyzer = APICoverageAnalyzer()

        # Test with function that has type hints
        def func_with_hints(x: int) -> int:
            return x * 2

        self.assertTrue(analyzer._has_type_hints(func_with_hints))

        # Test with function without type hints
        def func_without_hints(x):
            return x * 2

        self.assertFalse(analyzer._has_type_hints(func_without_hints))

    def test_check_web_docs(self):
        """Test web documentation checking"""
        temp_dir = tempfile.mkdtemp()
        doc_root = Path(temp_dir)
        doc_root.mkdir(exist_ok=True)
        (doc_root / 'generated').mkdir(exist_ok=True)

        analyzer = APICoverageAnalyzer(doc_root)

        # Create a fake documentation file
        (doc_root / 'generated' / 'torch.tensor.rst').write_text('content')

        # Should find the documentation
        has_docs = analyzer._check_web_docs('torch.tensor')
        self.assertTrue(has_docs)

        # Should not find non-existent documentation
        has_docs = analyzer._check_web_docs('torch.nonexistent')
        self.assertFalse(has_docs)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_tests()
