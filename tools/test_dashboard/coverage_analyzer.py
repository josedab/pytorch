#!/usr/bin/env python3
"""
Coverage Analyzer for OpInfo V2

This tool analyzes test coverage across all operators and generates reports
identifying coverage gaps and statistics.

Usage:
    python tools/test_dashboard/coverage_analyzer.py --op-db path/to/opinfo_db.py
"""

import argparse
import json
import sys
from typing import List, Dict, Set, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path

# Add PyTorch to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch.testing._internal.opinfo_v2 import OpInfoV2, TestCategory


@dataclass
class CoverageGap:
    """Represents a coverage gap for an operator.

    Attributes:
        operator: Name of the operator
        category: Test category that's missing
        severity: Severity level (critical, high, medium, low)
        reason: Human-readable description of the gap
    """
    operator: str
    category: str
    severity: str
    reason: str

    def __repr__(self):
        return f"{self.severity.upper()}: {self.operator} - {self.reason}"


@dataclass
class OperatorCoverage:
    """Coverage information for a single operator.

    Attributes:
        name: Operator name
        has_forward: Has forward tests
        has_backward: Has backward tests
        has_gradgrad: Has gradgrad tests
        has_forward_ad: Has forward AD tests
        has_jit: Has JIT tests
        has_vmap: Has vmap tests
        has_decomposition: Has decomposition tests
        has_serialization: Has serialization tests
        has_dtype_promotion: Has dtype promotion tests
        has_memory_format: Has memory format tests
        coverage_percentage: Overall coverage percentage
    """
    name: str
    has_forward: bool = False
    has_backward: bool = False
    has_gradgrad: bool = False
    has_forward_ad: bool = False
    has_jit: bool = False
    has_vmap: bool = False
    has_decomposition: bool = False
    has_serialization: bool = False
    has_dtype_promotion: bool = False
    has_memory_format: bool = False
    coverage_percentage: float = 0.0

    def calculate_coverage(self):
        """Calculate coverage percentage."""
        total_categories = 10
        covered = sum([
            self.has_forward,
            self.has_backward,
            self.has_gradgrad,
            self.has_forward_ad,
            self.has_jit,
            self.has_vmap,
            self.has_decomposition,
            self.has_serialization,
            self.has_dtype_promotion,
            self.has_memory_format,
        ])
        self.coverage_percentage = 100.0 * covered / total_categories


@dataclass
class CoverageReport:
    """Overall coverage report.

    Attributes:
        total_operators: Total number of operators
        operators: Per-operator coverage information
        gaps: List of coverage gaps
        statistics: Overall statistics
    """
    total_operators: int = 0
    operators: List[OperatorCoverage] = None
    gaps: List[CoverageGap] = None
    statistics: Dict[str, Any] = None

    def __post_init__(self):
        if self.operators is None:
            self.operators = []
        if self.gaps is None:
            self.gaps = []
        if self.statistics is None:
            self.statistics = {}


class CoverageAnalyzer:
    """Analyze test coverage across all operators.

    The CoverageAnalyzer examines OpInfoV2 definitions to:
    - Calculate coverage percentages
    - Identify operators missing critical tests
    - Generate reports and dashboards
    - Track coverage over time

    Example:
        >>> analyzer = CoverageAnalyzer()
        >>> report = analyzer.generate_report(op_db_v2)
        >>> analyzer.print_report(report)
        >>> analyzer.save_report(report, 'coverage_report.json')
    """

    # Critical test categories that every operator should have
    CRITICAL_CATEGORIES = {
        TestCategory.FORWARD,
        TestCategory.JIT,
    }

    # Test categories for differentiable operators
    GRADIENT_CATEGORIES = {
        TestCategory.BACKWARD,
        TestCategory.GRADGRAD,
    }

    def __init__(self):
        """Initialize the coverage analyzer."""
        pass

    def generate_report(self, op_db: List[OpInfoV2]) -> CoverageReport:
        """Generate coverage report from operator database.

        Args:
            op_db: List of OpInfoV2 definitions

        Returns:
            CoverageReport with detailed coverage information
        """
        report = CoverageReport()
        report.total_operators = len(op_db)

        # Analyze each operator
        for opinfo in op_db:
            coverage = self._analyze_operator(opinfo)
            report.operators.append(coverage)

        # Identify coverage gaps
        report.gaps = self._identify_gaps(op_db)

        # Calculate statistics
        report.statistics = self._calculate_statistics(report)

        return report

    def _analyze_operator(self, opinfo: OpInfoV2) -> OperatorCoverage:
        """Analyze coverage for a single operator.

        Args:
            opinfo: OpInfoV2 definition

        Returns:
            OperatorCoverage with coverage information
        """
        coverage = OperatorCoverage(name=opinfo.full_name)

        # Check which categories are covered
        coverage.has_forward = TestCategory.FORWARD in opinfo.test_categories
        coverage.has_backward = TestCategory.BACKWARD in opinfo.test_categories
        coverage.has_gradgrad = TestCategory.GRADGRAD in opinfo.test_categories
        coverage.has_forward_ad = TestCategory.FORWARD_AD in opinfo.test_categories
        coverage.has_jit = TestCategory.JIT in opinfo.test_categories
        coverage.has_vmap = TestCategory.VMAP in opinfo.test_categories
        coverage.has_decomposition = TestCategory.DECOMPOSITION in opinfo.test_categories
        coverage.has_serialization = TestCategory.SERIALIZATION in opinfo.test_categories
        coverage.has_dtype_promotion = TestCategory.DTYPE_PROMOTION in opinfo.test_categories
        coverage.has_memory_format = TestCategory.MEMORY_FORMAT in opinfo.test_categories

        coverage.calculate_coverage()

        return coverage

    def _identify_gaps(self, op_db: List[OpInfoV2]) -> List[CoverageGap]:
        """Identify coverage gaps across all operators.

        Args:
            op_db: List of OpInfoV2 definitions

        Returns:
            List of identified coverage gaps
        """
        gaps = []

        for opinfo in op_db:
            # Check for critical missing tests
            if TestCategory.FORWARD not in opinfo.test_categories:
                gaps.append(CoverageGap(
                    operator=opinfo.full_name,
                    category='FORWARD',
                    severity='critical',
                    reason='Missing forward execution tests'
                ))

            if TestCategory.JIT not in opinfo.test_categories:
                gaps.append(CoverageGap(
                    operator=opinfo.full_name,
                    category='JIT',
                    severity='critical',
                    reason='Missing TorchScript/JIT tests'
                ))

            # Check gradient tests for differentiable operators
            if opinfo.supports_autograd:
                if TestCategory.BACKWARD not in opinfo.test_categories:
                    gaps.append(CoverageGap(
                        operator=opinfo.full_name,
                        category='BACKWARD',
                        severity='high',
                        reason='Missing backward/gradient tests for differentiable op'
                    ))

                if opinfo.supports_gradgrad and TestCategory.GRADGRAD not in opinfo.test_categories:
                    gaps.append(CoverageGap(
                        operator=opinfo.full_name,
                        category='GRADGRAD',
                        severity='medium',
                        reason='Missing second-order gradient tests'
                    ))

            # Check vmap for operators that should support it
            if TestCategory.VMAP not in opinfo.test_categories:
                gaps.append(CoverageGap(
                    operator=opinfo.full_name,
                    category='VMAP',
                    severity='low',
                    reason='Missing vmap tests'
                ))

        return gaps

    def _calculate_statistics(self, report: CoverageReport) -> Dict[str, Any]:
        """Calculate overall statistics.

        Args:
            report: Coverage report

        Returns:
            Dictionary of statistics
        """
        stats = {}

        # Overall coverage
        if report.operators:
            avg_coverage = sum(op.coverage_percentage for op in report.operators) / len(report.operators)
            stats['average_coverage_percentage'] = round(avg_coverage, 2)
        else:
            stats['average_coverage_percentage'] = 0.0

        # Category coverage
        category_counts = defaultdict(int)
        for op in report.operators:
            if op.has_forward:
                category_counts['forward'] += 1
            if op.has_backward:
                category_counts['backward'] += 1
            if op.has_gradgrad:
                category_counts['gradgrad'] += 1
            if op.has_jit:
                category_counts['jit'] += 1
            if op.has_vmap:
                category_counts['vmap'] += 1
            if op.has_decomposition:
                category_counts['decomposition'] += 1
            if op.has_serialization:
                category_counts['serialization'] += 1
            if op.has_dtype_promotion:
                category_counts['dtype_promotion'] += 1
            if op.has_memory_format:
                category_counts['memory_format'] += 1
            if op.has_forward_ad:
                category_counts['forward_ad'] += 1

        total_ops = report.total_operators
        for category, count in category_counts.items():
            pct = 100.0 * count / total_ops if total_ops > 0 else 0.0
            stats[f'{category}_coverage_percentage'] = round(pct, 2)

        # Gap statistics
        gap_severity_counts = defaultdict(int)
        for gap in report.gaps:
            gap_severity_counts[gap.severity] += 1

        stats['gaps_by_severity'] = dict(gap_severity_counts)
        stats['total_gaps'] = len(report.gaps)

        return stats

    def print_report(self, report: CoverageReport, verbose: bool = False):
        """Print coverage report to console.

        Args:
            report: Coverage report
            verbose: Whether to print detailed per-operator coverage
        """
        print("=" * 80)
        print("OpInfo V2 Coverage Report")
        print("=" * 80)
        print()

        # Overall statistics
        print("Overall Statistics:")
        print(f"  Total Operators: {report.total_operators}")
        print(f"  Average Coverage: {report.statistics['average_coverage_percentage']:.1f}%")
        print(f"  Total Gaps: {report.statistics['total_gaps']}")
        print()

        # Category coverage
        print("Coverage by Category:")
        for category in ['forward', 'backward', 'gradgrad', 'jit', 'vmap']:
            key = f'{category}_coverage_percentage'
            if key in report.statistics:
                print(f"  {category.title()}: {report.statistics[key]:.1f}%")
        print()

        # Gaps by severity
        print("Gaps by Severity:")
        for severity in ['critical', 'high', 'medium', 'low']:
            count = report.statistics['gaps_by_severity'].get(severity, 0)
            print(f"  {severity.title()}: {count}")
        print()

        # Critical gaps
        critical_gaps = [g for g in report.gaps if g.severity == 'critical']
        if critical_gaps:
            print("Critical Gaps (showing first 10):")
            for gap in critical_gaps[:10]:
                print(f"  - {gap.operator}: {gap.reason}")
            if len(critical_gaps) > 10:
                print(f"  ... and {len(critical_gaps) - 10} more")
            print()

        # Operators with low coverage
        if verbose:
            print("Operators with <50% Coverage:")
            low_coverage = [op for op in report.operators if op.coverage_percentage < 50]
            low_coverage.sort(key=lambda x: x.coverage_percentage)
            for op in low_coverage[:20]:
                print(f"  {op.name}: {op.coverage_percentage:.1f}%")
            if len(low_coverage) > 20:
                print(f"  ... and {len(low_coverage) - 20} more")
            print()

    def save_report(self, report: CoverageReport, output_path: str):
        """Save coverage report to JSON file.

        Args:
            report: Coverage report
            output_path: Path to output JSON file
        """
        # Convert to dict
        report_dict = {
            'total_operators': report.total_operators,
            'operators': [asdict(op) for op in report.operators],
            'gaps': [asdict(gap) for gap in report.gaps],
            'statistics': report.statistics,
        }

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        print(f"Report saved to {output_path}")

    def generate_html_report(self, report: CoverageReport, output_path: str):
        """Generate HTML dashboard.

        Args:
            report: Coverage report
            output_path: Path to output HTML file
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OpInfo V2 Coverage Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-box {{
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            flex: 1;
        }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .gap-critical {{ color: #dc3545; }}
        .gap-high {{ color: #fd7e14; }}
        .gap-medium {{ color: #ffc107; }}
        .gap-low {{ color: #6c757d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background-color: #007bff;
        }}
    </style>
</head>
<body>
    <h1>OpInfo V2 Coverage Dashboard</h1>

    <div class="stats">
        <div class="stat-box">
            <div>Total Operators</div>
            <div class="stat-value">{report.total_operators}</div>
        </div>
        <div class="stat-box">
            <div>Average Coverage</div>
            <div class="stat-value">{report.statistics['average_coverage_percentage']:.1f}%</div>
        </div>
        <div class="stat-box">
            <div>Total Gaps</div>
            <div class="stat-value">{report.statistics['total_gaps']}</div>
        </div>
    </div>

    <h2>Coverage by Category</h2>
    <table>
        <tr>
            <th>Category</th>
            <th>Coverage</th>
            <th>Progress</th>
        </tr>
"""

        for category in ['forward', 'backward', 'gradgrad', 'jit', 'vmap']:
            key = f'{category}_coverage_percentage'
            if key in report.statistics:
                pct = report.statistics[key]
                html += f"""
        <tr>
            <td>{category.title()}</td>
            <td>{pct:.1f}%</td>
            <td>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {pct}%"></div>
                </div>
            </td>
        </tr>
"""

        html += """
    </table>

    <h2>Critical Gaps</h2>
    <table>
        <tr>
            <th>Operator</th>
            <th>Category</th>
            <th>Reason</th>
        </tr>
"""

        critical_gaps = [g for g in report.gaps if g.severity == 'critical']
        for gap in critical_gaps[:50]:
            html += f"""
        <tr>
            <td>{gap.operator}</td>
            <td>{gap.category}</td>
            <td>{gap.reason}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"HTML report saved to {output_path}")


def main():
    """Main entry point for coverage analyzer tool."""
    parser = argparse.ArgumentParser(
        description='Analyze test coverage for PyTorch operators'
    )
    parser.add_argument(
        '--op-db',
        type=str,
        required=True,
        help='Path to Python file containing op_db_v2 list'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output JSON report (optional)'
    )
    parser.add_argument(
        '--html',
        type=str,
        help='Path to output HTML report (optional)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed per-operator coverage'
    )

    args = parser.parse_args()

    # Load operator database
    import importlib.util
    spec = importlib.util.spec_from_file_location("opdb", args.op_db)
    opdb_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opdb_module)

    if not hasattr(opdb_module, 'op_db_v2'):
        print("Error: op_db_v2 not found in module")
        sys.exit(1)

    op_db = opdb_module.op_db_v2

    # Generate report
    analyzer = CoverageAnalyzer()
    report = analyzer.generate_report(op_db)

    # Print to console
    analyzer.print_report(report, verbose=args.verbose)

    # Save JSON if requested
    if args.output:
        analyzer.save_report(report, args.output)

    # Generate HTML if requested
    if args.html:
        analyzer.generate_html_report(report, args.html)


if __name__ == '__main__':
    main()
