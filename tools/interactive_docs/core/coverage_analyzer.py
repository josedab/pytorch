"""
API Coverage Analyzer

This module analyzes documentation coverage of the PyTorch API,
tracking which APIs have documentation, examples, type hints, etc.

Based on RFC-0004: Interactive Documentation with Live Examples
"""

import inspect
import importlib
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import re


@dataclass
class DocStatus:
    """Documentation status for an API"""
    has_docstring: bool
    has_examples: bool
    has_type_hints: bool
    has_web_docs: bool
    docstring_length: int = 0
    num_examples: int = 0


@dataclass
class CoverageReport:
    """Overall API coverage report"""
    total_apis: int
    documented: int
    has_examples: int
    has_type_hints: int
    coverage_by_module: Dict[str, 'ModuleCoverage']


@dataclass
class ModuleCoverage:
    """Coverage statistics for a module"""
    total: int
    documented: int
    has_examples: int
    has_type_hints: int

    @property
    def doc_percentage(self) -> float:
        """Percentage of APIs with documentation"""
        return 100 * self.documented / self.total if self.total > 0 else 0

    @property
    def example_percentage(self) -> float:
        """Percentage of APIs with examples"""
        return 100 * self.has_examples / self.total if self.total > 0 else 0


class APICoverageAnalyzer:
    """Analyze documentation coverage of PyTorch API"""

    def __init__(self, doc_root: Optional[Path] = None):
        """
        Initialize the coverage analyzer.

        Args:
            doc_root: Root directory of web documentation (optional)
        """
        self.doc_root = Path(doc_root) if doc_root else None
        self.web_docs_cache = {}

    def discover_public_apis(self, module_name: str = 'torch') -> List[str]:
        """
        Find all public APIs in a module.

        Args:
            module_name: Name of the module to analyze

        Returns:
            List of fully qualified API names
        """
        apis = []

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return apis

        # Scan module for public APIs
        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue

            if callable(obj) or inspect.isclass(obj):
                apis.append(f'{module_name}.{name}')

        # Scan important submodules
        submodules = self._get_submodules(module_name)
        for submodule_name in submodules:
            try:
                submodule = importlib.import_module(f'{module_name}.{submodule_name}')
                for name, obj in inspect.getmembers(submodule):
                    if name.startswith('_'):
                        continue

                    if callable(obj) or inspect.isclass(obj):
                        full_name = f'{module_name}.{submodule_name}.{name}'
                        apis.append(full_name)

                        # For torch.nn, also check functional
                        if submodule_name == 'nn' and hasattr(submodule, 'functional'):
                            functional = submodule.functional
                            for func_name, func_obj in inspect.getmembers(functional):
                                if func_name.startswith('_'):
                                    continue
                                if callable(func_obj):
                                    apis.append(f'{module_name}.nn.functional.{func_name}')

            except (ImportError, AttributeError):
                continue

        return sorted(set(apis))

    def _get_submodules(self, module_name: str) -> List[str]:
        """
        Get list of important submodules to analyze.

        Args:
            module_name: Name of the parent module

        Returns:
            List of submodule names
        """
        if module_name == 'torch':
            return [
                'nn',
                'optim',
                'autograd',
                'cuda',
                'distributed',
                'jit',
                'onnx',
                'utils',
                'sparse',
                'special',
                'fft',
                'linalg',
            ]
        return []

    def check_documentation(self, api: str) -> DocStatus:
        """
        Check documentation status of an API.

        Args:
            api: Fully qualified API name (e.g., 'torch.nn.ReLU')

        Returns:
            Documentation status
        """
        obj = self._import_api(api)
        if obj is None:
            return DocStatus(
                has_docstring=False,
                has_examples=False,
                has_type_hints=False,
                has_web_docs=False,
            )

        # Check docstring
        docstring = inspect.getdoc(obj)
        has_docstring = docstring is not None and len(docstring) > 50
        docstring_length = len(docstring) if docstring else 0

        # Check for examples in docstring
        has_examples = False
        num_examples = 0
        if docstring:
            # Look for doctest examples (>>>)
            if '>>>' in docstring:
                has_examples = True
                num_examples = docstring.count('>>>')

            # Look for code blocks
            if '.. code-block::' in docstring or '```python' in docstring:
                has_examples = True
                num_examples += len(re.findall(r'(.. code-block::|```python)', docstring))

        # Check for type hints
        has_type_hints = self._has_type_hints(obj)

        # Check for web documentation
        has_web_docs = self._check_web_docs(api)

        return DocStatus(
            has_docstring=has_docstring,
            has_examples=has_examples,
            has_type_hints=has_type_hints,
            has_web_docs=has_web_docs,
            docstring_length=docstring_length,
            num_examples=num_examples,
        )

    def _import_api(self, api: str) -> Optional[Any]:
        """
        Import an API by its fully qualified name.

        Args:
            api: Fully qualified API name

        Returns:
            The imported object, or None if import fails
        """
        parts = api.split('.')
        try:
            # Try to import as module first
            module = importlib.import_module('.'.join(parts[:-1]))
            return getattr(module, parts[-1])
        except (ImportError, AttributeError):
            try:
                # Try importing more levels
                if len(parts) > 2:
                    module = importlib.import_module('.'.join(parts[:-2]))
                    obj = getattr(module, parts[-2])
                    return getattr(obj, parts[-1])
            except (ImportError, AttributeError):
                pass
        return None

    def _has_type_hints(self, obj: Any) -> bool:
        """
        Check if an object has type hints.

        Args:
            obj: Object to check

        Returns:
            True if object has type hints
        """
        try:
            # Check if it's a callable with annotations
            if callable(obj):
                sig = inspect.signature(obj)
                # Check if any parameter or return has annotation
                if sig.return_annotation != inspect.Signature.empty:
                    return True
                for param in sig.parameters.values():
                    if param.annotation != inspect.Parameter.empty:
                        return True
        except (ValueError, TypeError):
            pass
        return False

    def _check_web_docs(self, api: str) -> bool:
        """
        Check if API has web documentation.

        Args:
            api: Fully qualified API name

        Returns:
            True if web documentation exists
        """
        if self.doc_root is None:
            return False

        # Check cache
        if api in self.web_docs_cache:
            return self.web_docs_cache[api]

        # Convert API name to expected documentation path
        # e.g., torch.nn.ReLU -> generated/torch.nn.ReLU.rst
        doc_path = self.doc_root / 'generated' / f'{api}.rst'
        exists = doc_path.exists()

        # Also check in other common locations
        if not exists:
            # Check in module index
            parts = api.split('.')
            if len(parts) >= 2:
                module_doc = self.doc_root / f'{parts[1]}.rst'
                if module_doc.exists():
                    with open(module_doc) as f:
                        content = f.read()
                        # Check if API is mentioned
                        exists = api in content or parts[-1] in content

        self.web_docs_cache[api] = exists
        return exists

    def analyze_coverage(self, module_name: str = 'torch') -> CoverageReport:
        """
        Generate API coverage report.

        Args:
            module_name: Name of module to analyze

        Returns:
            Coverage report
        """
        # Get all public APIs
        all_apis = self.discover_public_apis(module_name)

        # Check documentation status
        coverage = {}
        for api in all_apis:
            status = self.check_documentation(api)
            coverage[api] = status

        # Calculate overall statistics
        total_apis = len(all_apis)
        documented = sum(1 for s in coverage.values() if s.has_docstring)
        has_examples = sum(1 for s in coverage.values() if s.has_examples)
        has_type_hints = sum(1 for s in coverage.values() if s.has_type_hints)

        # Group by module
        coverage_by_module = self._group_by_module(coverage)

        return CoverageReport(
            total_apis=total_apis,
            documented=documented,
            has_examples=has_examples,
            has_type_hints=has_type_hints,
            coverage_by_module=coverage_by_module,
        )

    def _group_by_module(self, coverage: Dict[str, DocStatus]) -> Dict[str, ModuleCoverage]:
        """
        Group coverage statistics by module.

        Args:
            coverage: API coverage dictionary

        Returns:
            Dictionary mapping module names to their coverage
        """
        module_stats = {}

        for api, status in coverage.items():
            # Extract module name (e.g., 'torch.nn.ReLU' -> 'torch.nn')
            parts = api.split('.')
            if len(parts) >= 2:
                module = '.'.join(parts[:2])
            else:
                module = parts[0]

            if module not in module_stats:
                module_stats[module] = {
                    'total': 0,
                    'documented': 0,
                    'has_examples': 0,
                    'has_type_hints': 0,
                }

            module_stats[module]['total'] += 1
            if status.has_docstring:
                module_stats[module]['documented'] += 1
            if status.has_examples:
                module_stats[module]['has_examples'] += 1
            if status.has_type_hints:
                module_stats[module]['has_type_hints'] += 1

        return {
            name: ModuleCoverage(**stats)
            for name, stats in module_stats.items()
        }

    def generate_report(self, report: CoverageReport) -> str:
        """
        Generate a human-readable coverage report.

        Args:
            report: Coverage report from analyze_coverage

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            "PyTorch API Documentation Coverage Report",
            "=" * 70,
            "",
            f"Overall: {100 * report.documented / report.total_apis:.0f}% "
            f"({report.documented} / {report.total_apis} APIs documented)",
            "",
            "Detailed Statistics:",
            f"  APIs with documentation: {report.documented} / {report.total_apis} "
            f"({100 * report.documented / report.total_apis:.1f}%)",
            f"  APIs with examples:      {report.has_examples} / {report.total_apis} "
            f"({100 * report.has_examples / report.total_apis:.1f}%)",
            f"  APIs with type hints:    {report.has_type_hints} / {report.total_apis} "
            f"({100 * report.has_type_hints / report.total_apis:.1f}%)",
            "",
            "Coverage by Module:",
            "-" * 70,
        ]

        # Sort modules by documentation percentage
        sorted_modules = sorted(
            report.coverage_by_module.items(),
            key=lambda x: x[1].doc_percentage,
            reverse=True
        )

        for module_name, coverage in sorted_modules:
            bar_length = 20
            filled = int(bar_length * coverage.doc_percentage / 100)
            bar = '█' * filled + '░' * (bar_length - filled)

            lines.append(
                f"  {module_name:30s} {bar} {coverage.doc_percentage:5.1f}% "
                f"({coverage.documented} / {coverage.total})"
            )

        lines.append("")
        lines.append("=" * 70)

        return '\n'.join(lines)

    def find_missing_examples(self, report: CoverageReport,
                            limit: int = 50) -> List[str]:
        """
        Find APIs that are missing examples.

        Args:
            report: Coverage report
            limit: Maximum number of APIs to return

        Returns:
            List of API names missing examples
        """
        missing = []
        for api, status in report.coverage_by_module.items():
            # This is a simplified version - would need to track individual APIs
            pass
        return missing[:limit]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze PyTorch API documentation coverage"
    )
    parser.add_argument(
        '--module',
        default='torch',
        help='Module to analyze (default: torch)'
    )
    parser.add_argument(
        '--doc-root',
        type=Path,
        help='Root directory of web documentation'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )

    args = parser.parse_args()

    analyzer = APICoverageAnalyzer(args.doc_root)
    report = analyzer.analyze_coverage(args.module)

    if args.json:
        import json
        output = {
            'total_apis': report.total_apis,
            'documented': report.documented,
            'has_examples': report.has_examples,
            'has_type_hints': report.has_type_hints,
            'by_module': {
                name: {
                    'total': cov.total,
                    'documented': cov.documented,
                    'has_examples': cov.has_examples,
                    'has_type_hints': cov.has_type_hints,
                    'doc_percentage': cov.doc_percentage,
                }
                for name, cov in report.coverage_by_module.items()
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print(analyzer.generate_report(report))
