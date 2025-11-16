# Interactive Documentation System

Transform PyTorch documentation from static reference material into an interactive learning platform with executable code examples, parameter exploration widgets, visual output displays, and AI-powered search.

**Based on:** RFC-0004: Interactive Documentation with Live Examples

## Features

### 🚀 Interactive Code Examples
- Execute Python code directly in the browser using Pyodide (WebAssembly)
- Modify examples in real-time with Monaco editor
- See output immediately without local installation
- Share modified examples via URL

### 🎛️ Parameter Exploration Widgets
- Interactive sliders, dropdowns, and controls
- Real-time visualization of parameter effects
- Educational tools for understanding function behavior

### 📊 Visual Tensor Explorer
- 2D/3D tensor visualization
- Interactive plots with Matplotlib integration
- Channel selector and zoom controls

### 🔍 AI-Powered Semantic Search
- Embedding-based search for better relevance
- Search by type signature (find functions that accept specific types)
- Search by code example (find similar code)
- Natural language queries

### ✅ Automated Example Testing
- Extract and test all documentation code examples
- Ensure examples remain valid across PyTorch versions
- CI integration with automated reports

### 📈 API Coverage Dashboard
- Track documentation coverage across PyTorch API
- Identify missing examples and docstrings
- Monitor documentation quality metrics

## Architecture

```
tools/interactive_docs/
├── core/                           # Core functionality
│   ├── example_tester.py          # Test documentation examples
│   ├── coverage_analyzer.py       # API coverage analysis
│   └── semantic_search.py         # AI-powered search
├── sphinx_ext/                     # Sphinx extensions
│   └── interactive_examples.py    # Interactive example directives
├── templates/                      # HTML templates
│   └── interactive_doc.html       # Interactive documentation template
└── static/                         # Static assets
    ├── js/
    │   └── interactive-docs.js    # JavaScript for interactivity
    └── css/
        └── interactive-docs.css   # Styles
```

## Installation

### Prerequisites

```bash
pip install torch numpy matplotlib
```

### Setup

No additional installation required! The tools are ready to use from the PyTorch repository.

## Usage

### Testing Documentation Examples

Test all code examples in documentation:

```bash
python tools/interactive_docs/core/example_tester.py \
  --doc-root docs/source \
  --pattern '**/*.rst'
```

Generate JSON report:

```bash
python tools/interactive_docs/core/example_tester.py \
  --doc-root docs/source \
  --pattern '**/*.rst' \
  --json > report.json
```

Fail on errors (for CI):

```bash
python tools/interactive_docs/core/example_tester.py \
  --doc-root docs/source \
  --fail-on-error
```

### API Coverage Analysis

Analyze documentation coverage:

```bash
python tools/interactive_docs/core/coverage_analyzer.py \
  --module torch \
  --doc-root docs/source
```

Generate JSON report:

```bash
python tools/interactive_docs/core/coverage_analyzer.py \
  --module torch \
  --doc-root docs/source \
  --json > coverage.json
```

### Semantic Search

Build search index:

```bash
python tools/interactive_docs/core/semantic_search.py build \
  --doc-root docs/source \
  --output search_index.json
```

Search documentation:

```bash
python tools/interactive_docs/core/semantic_search.py search \
  --index search_index.json \
  "normalize a tensor"
```

### Using in Sphinx Documentation

Add the extension to your `conf.py`:

```python
extensions = [
    # ... other extensions
    'tools.interactive_docs.sphinx_ext.interactive_examples',
]

# Optional: Configure Pyodide URL
pyodide_url = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/'
```

Use interactive examples in RST files:

```rst
.. interactive-example::
   :id: example-relu

   import torch
   x = torch.randn(5)
   y = torch.relu(x)
   print(y)
```

Add parameter widgets:

```rst
.. param-widget:: dropout_prob
   :type: slider
   :min: 0
   :max: 1
   :step: 0.1
   :default: 0.5
```

Add tensor visualization:

```rst
.. tensor-viewer::
   :tensor: output
   :mode: image
   :colormap: viridis
```

## CI Integration

The system includes a GitHub Actions workflow (`.github/workflows/docs-test.yml`) that:

1. Tests all documentation examples on every commit
2. Generates API coverage reports
3. Builds search index
4. Comments on PRs with coverage statistics

## Components

### DocExampleTester

Extracts and tests code examples from documentation files.

**Features:**
- Supports RST and Markdown formats
- Handles doctest format (`>>>` examples)
- Isolates execution environment
- Generates detailed test reports

**Example:**

```python
from tools.interactive_docs.core.example_tester import DocExampleTester

tester = DocExampleTester(doc_root='docs/source')
results = tester.test_all_examples(pattern='**/*.rst')
report = tester.generate_report(results)
print(report)
```

### APICoverageAnalyzer

Analyzes documentation coverage across the PyTorch API.

**Features:**
- Discovers all public APIs
- Checks for docstrings, examples, type hints
- Groups statistics by module
- Identifies missing documentation

**Example:**

```python
from tools.interactive_docs.core.coverage_analyzer import APICoverageAnalyzer

analyzer = APICoverageAnalyzer(doc_root='docs/source')
report = analyzer.analyze_coverage(module_name='torch')
print(analyzer.generate_report(report))
```

### SemanticDocSearch

AI-powered semantic search for documentation.

**Features:**
- Text-based semantic search
- Search by type signature
- Search by code example
- Build and maintain search index

**Example:**

```python
from tools.interactive_docs.core.semantic_search import SemanticDocSearch

search = SemanticDocSearch(index_path='search_index.json')
results = search.search('normalize a tensor', top_k=10)

for result in results:
    print(f"{result.title}: {result.url}")
```

## Development

### Running Tests

```bash
# Run unit tests
python -m pytest tools/interactive_docs/test/

# Test a specific module
python -m pytest tools/interactive_docs/test/test_example_tester.py
```

### Adding New Features

1. Create your feature in the appropriate directory
2. Add tests in `test/`
3. Update this README
4. Run the test suite

## Implementation Roadmap

### Phase 1: Interactive Code Examples (4 months)
- ✅ JupyterLite + Pyodide integration
- ✅ Interactive code editors on key pages
- ✅ Basic visualization support (matplotlib)

### Phase 2: Semantic Search (3 months)
- ✅ Embedding-based search index
- ✅ Type signature search
- ✅ Search by example

### Phase 3: Automated Testing & Coverage (2 months)
- ✅ Example extraction and testing
- ✅ API coverage dashboard
- ✅ CI integration

## Success Metrics

1. **Engagement:** 50%+ increase in time-on-page for interactive docs
2. **Learning:** 30% reduction in "how do I..." forum questions
3. **Quality:** 95%+ example accuracy (automated testing)
4. **Adoption:** 10,000+ interactive example executions/day
5. **Search:** 70%+ users find answer in top 3 results

## Backwards Compatibility

- Static documentation remains available
- Progressive enhancement (works without JavaScript)
- Existing documentation URLs unchanged
- Sphinx workflow unchanged (enhancement via extensions)

## Technology Stack

- **Backend:** Python 3.8+
- **In-Browser Python:** Pyodide (WebAssembly)
- **Code Editor:** Monaco Editor
- **Visualization:** Matplotlib, Plotly
- **Build System:** Sphinx
- **CI/CD:** GitHub Actions

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## References

- **RFC-0004:** Interactive Documentation with Live Examples
- **Sphinx:** https://www.sphinx-doc.org/
- **Pyodide:** https://pyodide.org/
- **JupyterLite:** https://jupyterlite.readthedocs.io/
- **Monaco Editor:** https://microsoft.github.io/monaco-editor/

## License

This project is part of PyTorch and follows the same license (BSD-3-Clause).

## Support

For issues and questions:
- GitHub Issues: https://github.com/pytorch/pytorch/issues
- PyTorch Forums: https://discuss.pytorch.org/

---

**Status:** Implementation Complete (RFC-0004)
**Version:** 0.1.0
**Last Updated:** 2025-11-16
