# RFC-0004: Interactive Documentation with Live Examples

**Status:** Draft
**Author:** Claude Code Analysis
**Created:** 2025-11-16
**Based on commit:** [`5d99a795f54d6bf14e39ae12df58d760d4fd8984`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984)

## Summary

Transform PyTorch documentation from static reference material into an interactive learning platform with executable code examples, parameter exploration widgets, visual output displays, and AI-powered search. This significantly improves the developer experience and reduces time-to-productivity for new users.

## Motivation

### Current Documentation Limitations

The existing documentation system (built with Sphinx, configured in [`docs/source/conf.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/docs/source/conf.py)) is comprehensive but has several usability issues:

1. **Static Examples:**
   ```python
   # From documentation
   >>> x = torch.randn(3, 4)
   >>> y = torch.relu(x)
   ```
   - Can't modify parameters interactively
   - Can't see output without running locally
   - No visualization of results

2. **Fragmented Information:**
   - API reference separate from tutorials
   - Examples scattered across different pages
   - Hard to find "how do I..." answers
   - No connection between related concepts

3. **Limited Search:**
   - Keyword-based search only
   - No semantic understanding ("normalize a tensor" doesn't find `F.normalize`)
   - Can't search by example
   - No type-based search ("find functions that accept Tensor and return Tensor")

4. **No Verification:**
   - Examples can become outdated
   - No automatic testing of documentation code
   - Broken links go unnoticed

### User Pain Points

**Beginners:**
- "I want to try code examples without installing PyTorch"
- "I don't understand what this parameter does"
- "How do I visualize this tensor?"

**Experienced Users:**
- "What's the difference between these similar functions?"
- "Which is faster for my use case?"
- "Show me all functions that work with sparse tensors"

## Detailed Design

### Interactive Documentation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Interactive Documentation                   │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Web)                                             │
│  ├─ Interactive code editor (Monaco/CodeMirror)             │
│  ├─ Parameter widgets (sliders, dropdowns)                  │
│  ├─ Output visualization (matplotlib, 3D tensor viewer)     │
│  └─ AI-powered search                                       │
├─────────────────────────────────────────────────────────────┤
│  Backend (JupyterLite / Pyodide)                            │
│  ├─ In-browser PyTorch execution                            │
│  ├─ Code sandboxing                                         │
│  └─ Result caching                                          │
├─────────────────────────────────────────────────────────────┤
│  Documentation Generation                                   │
│  ├─ Enhanced Sphinx extensions                              │
│  ├─ Automatic example extraction                            │
│  ├─ Type signature indexing                                 │
│  └─ Cross-reference resolution                              │
├─────────────────────────────────────────────────────────────┤
│  Testing & Validation                                       │
│  ├─ Automated example testing                               │
│  ├─ Link checking                                           │
│  └─ API coverage analysis                                   │
└─────────────────────────────────────────────────────────────┘
```

### Component 1: Interactive Code Examples

**Technology:** JupyterLite (in-browser Jupyter) + Pyodide (Python in WebAssembly)

**Example Enhancement:**

```markdown
<!-- Current documentation -->
## torch.relu

```python
>>> x = torch.randn(5)
>>> y = torch.relu(x)
```

<!-- Enhanced interactive documentation -->
## torch.relu

<interactive-example>
```python
import torch
import matplotlib.pyplot as plt

# Try modifying these values!
x = torch.randn(100)

# Apply ReLU
y = torch.relu(x)

# Visualize
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(x.numpy(), bins=20)
plt.title('Input Distribution')

plt.subplot(1, 2, 2)
plt.hist(y.numpy(), bins=20)
plt.title('After ReLU')
plt.show()
```
</interactive-example>

**Parameters:**
- `input` (Tensor): <param-widget type="tensor" default="torch.randn(5)" />

  Try different inputs:
  - <preset>torch.linspace(-5, 5, 100)</preset>
  - <preset>torch.tensor([-1, 0, 1, 2, 3])</preset>
  - <preset>torch.randn(3, 4)</preset>
```

**User Experience:**
1. Click "Run" to execute code
2. Modify code directly in browser
3. Adjust parameters with widgets
4. See output immediately (text, plots, tensor values)
5. Share modified examples via URL

### Component 2: Parameter Exploration Widgets

**For functions with numeric parameters:**

```markdown
## torch.nn.functional.dropout

**Interactive Playground:**

<param-explorer>
```python
input = torch.randn(1000)
output = F.dropout(input, p=<slider min=0 max=1 step=0.1 default=0.5 />)

print(f"Zeros: {(output == 0).sum().item()} / {output.numel()}")
print(f"Mean: {output.mean():.4f} (expected: {input.mean():.4f})")
```
</param-explorer>

**Impact of `p` parameter:**
<visualization type="chart">
  x-axis: dropout probability (0 to 1)
  y-axis: percentage of zeros in output
  interactive: drag slider to update chart
</visualization>
```

### Component 3: Visual Tensor Explorer

**3D/2D tensor visualization:**

```markdown
## torch.nn.functional.conv2d

<tensor-viewer>
```python
# Input image
input = torch.randn(1, 3, 28, 28)  # [batch, channels, height, width]

# Convolution kernel
kernel = torch.randn(16, 3, 3, 3)  # [out_channels, in_channels, kH, kW]

# Apply convolution
output = F.conv2d(input, kernel, padding=1)
```

**Visualization:**
- Input: <tensor-display tensor="input" mode="image" colormap="viridis" />
- Kernel: <tensor-display tensor="kernel[0]" mode="filters" />
- Output: <tensor-display tensor="output" mode="image" colormap="viridis" />

**Controls:**
- Channel selector: <slider min=0 max=15 />
- Zoom: <zoom-widget />
</tensor-viewer>
```

### Component 4: AI-Powered Semantic Search

**Enhanced search with embedding-based retrieval:**

```python
class SemanticDocSearch:
    """AI-powered documentation search"""

    def __init__(self, doc_embeddings: Dict[str, np.ndarray]):
        self.embeddings = doc_embeddings
        self.doc_index = list(doc_embeddings.keys())
        self.embedding_matrix = np.vstack(list(doc_embeddings.values()))

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Semantic search for query"""
        # Embed query
        query_embedding = self._embed_text(query)

        # Cosine similarity with all docs
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embedding_matrix
        )[0]

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                title=self.doc_index[idx],
                url=self._get_url(self.doc_index[idx]),
                score=similarities[idx],
                snippet=self._get_snippet(self.doc_index[idx], query)
            ))

        return results

    def search_by_signature(self,
                           input_types: List[type],
                           output_type: type) -> List[FunctionInfo]:
        """Search functions by type signature"""
        # Example: "Find functions that take (Tensor, Tensor) -> Tensor"
        matches = []

        for func_name, sig in self.type_signatures.items():
            if (sig.input_types == input_types and
                sig.output_type == output_type):
                matches.append(FunctionInfo(name=func_name, signature=sig))

        return matches

    def search_by_example(self, code: str) -> List[SearchResult]:
        """Find similar code examples"""
        # Embed code
        code_embedding = self._embed_code(code)

        # Search in example embeddings
        # ... (similar to text search)
```

**Search Interface:**

```html
<!-- On pytorch.org -->
<search-bar>
  <input placeholder="Search documentation... (try 'normalize a tensor' or 'functions like F.relu')" />

  <search-modes>
    <tab>Text</tab>
    <tab>By Example</tab>
    <tab>By Signature</tab>
  </search-modes>
</search-bar>

<!-- Example: Search by signature -->
<signature-search>
  Input types: <type-selector>[Tensor, Tensor]</type-selector>
  Output type: <type-selector>Tensor</type-selector>
  <button>Search</button>

  Results:
  - torch.add(Tensor, Tensor) -> Tensor
  - torch.mul(Tensor, Tensor) -> Tensor
  - torch.matmul(Tensor, Tensor) -> Tensor
  - ... (50 more)
</signature-search>
```

### Component 5: Automated Example Testing

**Ensure all documentation code examples work:**

```python
# tools/docs/test_examples.py

class DocExampleTester:
    """Extract and test code examples from documentation"""

    def extract_examples(self, doc_file: Path) -> List[CodeExample]:
        """Extract code examples from .rst or .md file"""
        examples = []

        # Parse documentation
        with open(doc_file) as f:
            content = f.read()

        # Find all code blocks
        for match in re.finditer(r'```python\n(.*?)\n```', content, re.DOTALL):
            code = match.group(1)
            examples.append(CodeExample(
                file=doc_file,
                line=content[:match.start()].count('\n'),
                code=code
            ))

        return examples

    def test_example(self, example: CodeExample) -> TestResult:
        """Execute example and verify it works"""
        try:
            # Create isolated namespace
            namespace = {'torch': torch, 'np': np}

            # Execute code
            exec(example.code, namespace)

            # Verify expected outputs (if specified)
            if example.expected_output:
                self.verify_output(namespace, example.expected_output)

            return TestResult(success=True)

        except Exception as e:
            return TestResult(
                success=False,
                error=str(e),
                traceback=traceback.format_exc()
            )

    def test_all_examples(self) -> Dict[Path, List[TestResult]]:
        """Test all examples in documentation"""
        results = {}

        for doc_file in self.find_doc_files():
            examples = self.extract_examples(doc_file)
            results[doc_file] = [
                self.test_example(ex) for ex in examples
            ]

        return results

    def generate_report(self, results: Dict[Path, List[TestResult]]) -> str:
        """Generate test report"""
        total = sum(len(tests) for tests in results.values())
        passed = sum(
            sum(t.success for t in tests)
            for tests in results.values()
        )

        report = f"Documentation Example Tests: {passed}/{total} passed\n\n"

        for doc_file, tests in results.items():
            failures = [t for t in tests if not t.success]
            if failures:
                report += f"{doc_file}:\n"
                for failure in failures:
                    report += f"  - Line {failure.line}: {failure.error}\n"

        return report
```

**Integration with CI:**

```yaml
# .github/workflows/docs-test.yml
name: Test Documentation Examples

on: [push, pull_request]

jobs:
  test-examples:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install PyTorch
        run: pip install torch
      - name: Test documentation examples
        run: python tools/docs/test_examples.py --fail-on-error
```

### Component 6: API Coverage Dashboard

**Track documentation coverage:**

```python
class APICoverageAnalyzer:
    """Analyze documentation coverage of PyTorch API"""

    def analyze_coverage(self) -> CoverageReport:
        """Generate API coverage report"""
        # Get all public APIs
        all_apis = self.discover_public_apis()

        # Check documentation status
        coverage = {}
        for api in all_apis:
            status = self.check_documentation(api)
            coverage[api] = status

        return CoverageReport(
            total_apis=len(all_apis),
            documented=sum(1 for s in coverage.values() if s.has_docstring),
            has_examples=sum(1 for s in coverage.values() if s.has_examples),
            has_type_hints=sum(1 for s in coverage.values() if s.has_type_hints),
            coverage_by_module=self.group_by_module(coverage)
        )

    def discover_public_apis(self) -> List[str]:
        """Find all public APIs in PyTorch"""
        apis = []

        # Scan torch module
        for name, obj in inspect.getmembers(torch):
            if not name.startswith('_') and callable(obj):
                apis.append(f'torch.{name}')

        # Scan submodules (torch.nn, torch.optim, etc.)
        for submodule_name in ['nn', 'optim', 'autograd', 'cuda']:
            submodule = getattr(torch, submodule_name)
            for name, obj in inspect.getmembers(submodule):
                if not name.startswith('_') and callable(obj):
                    apis.append(f'torch.{submodule_name}.{name}')

        return apis

    def check_documentation(self, api: str) -> DocStatus:
        """Check documentation status of an API"""
        obj = self.import_api(api)

        return DocStatus(
            has_docstring=obj.__doc__ is not None and len(obj.__doc__) > 50,
            has_examples='>>>' in (obj.__doc__ or ''),
            has_type_hints=self.has_type_hints(obj),
            has_web_docs=self.check_web_docs(api),
        )
```

**Dashboard visualization:**

```
PyTorch API Documentation Coverage

Overall: ████████████░░░░░░░░ 62% (1,234 / 2,000 APIs documented)

By Module:
  torch.nn:        ████████████████░░░░ 82% (312 / 380)
  torch.optim:     ████████████████████ 95% (38 / 40)
  torch (root):    ████████░░░░░░░░░░░░ 45% (234 / 520)
  torch.autograd:  ██████████████░░░░░░ 70% (56 / 80)

Missing Examples (high priority):
  - torch.gather
  - torch.nn.functional.fold
  - torch.nn.InstanceNorm1d
  ... (50 more)
```

## Example: Complete Interactive Documentation Page

```html
<!-- pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html -->

<!DOCTYPE html>
<html>
<head>
  <title>torch.nn.functional.dropout — PyTorch Documentation</title>
  <script src="pyodide.js"></script>
  <script src="interactive-docs.js"></script>
</head>
<body>
  <h1>torch.nn.functional.dropout</h1>

  <div class="function-signature">
    <code>torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False) → Tensor</code>
  </div>

  <div class="description">
    <p>Randomly zero out elements of the input tensor during training using samples from a Bernoulli distribution.</p>
  </div>

  <h2>Parameters</h2>
  <dl>
    <dt>input (Tensor)</dt>
    <dd>Input tensor of any shape</dd>

    <dt>p (float)</dt>
    <dd>Probability of an element to be zeroed. Default: 0.5
      <br><em>Interactive:</em> <input type="range" min="0" max="1" step="0.05" value="0.5" id="dropout-p"/>
    </dd>

    <dt>training (bool)</dt>
    <dd>If True, apply dropout. If False, return input unchanged. Default: True</dd>

    <dt>inplace (bool)</dt>
    <dd>If True, modify input in-place. Default: False</dd>
  </dl>

  <h2>Interactive Example</h2>
  <interactive-code-editor id="example-1">
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Create input
x = torch.randn(1000)

# Apply dropout
y = F.dropout(x, p=0.5, training=True)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(x.numpy(), bins=30, alpha=0.7, label='Input')
ax1.hist(y.numpy(), bins=30, alpha=0.7, label='Output')
ax1.legend()
ax1.set_title('Value Distribution')

ax2.scatter(range(100), x[:100].numpy(), alpha=0.5, label='Input')
ax2.scatter(range(100), y[:100].numpy(), alpha=0.5, label='Output')
ax2.legend()
ax2.set_title('First 100 Values')

plt.tight_layout()
plt.show()

print(f"Dropout rate: {(y == 0).float().mean():.2%}")
print(f"Expected: {p:.2%}")
  </interactive-code-editor>

  <button onclick="runCode('example-1')">Run Example</button>
  <div id="output-1" class="code-output"></div>

  <h2>See Also</h2>
  <ul>
    <li><a href="dropout2d.html">torch.nn.functional.dropout2d</a> - 2D dropout</li>
    <li><a href="../nn/Dropout.html">torch.nn.Dropout</a> - Dropout layer</li>
    <li><a href="alpha_dropout.html">torch.nn.functional.alpha_dropout</a> - Alpha dropout for SELU</li>
  </ul>

  <h2>Related Tutorials</h2>
  <ul>
    <li><a href="/tutorials/beginner/basics/buildmodel_tutorial.html">Building Models</a></li>
    <li><a href="/tutorials/intermediate/regularization_tutorial.html">Regularization Techniques</a></li>
  </ul>
</body>
</html>
```

## Implementation Plan

### Phase 1: Interactive Code Examples (4 months)

**Deliverables:**
1. JupyterLite + Pyodide integration
2. Interactive code editors on 50 key pages
3. Basic visualization support (matplotlib)

**Success Criteria:**
- Examples run in < 2 seconds
- Works in all major browsers
- 1000+ page views/day on interactive examples

### Phase 2: Semantic Search (3 months)

**Deliverables:**
1. Embedding-based search index
2. Type signature search
3. Search by example

**Success Criteria:**
- 80%+ relevant results in top 5
- 50%+ faster time-to-find than keyword search
- 5000+ searches/day

### Phase 3: Automated Testing & Coverage (2 months)

**Deliverables:**
1. Example extraction and testing
2. API coverage dashboard
3. CI integration

**Success Criteria:**
- 95%+ examples pass automated tests
- Coverage dashboard updated nightly
- <1% broken examples escape to production

## Backwards Compatibility

- Static documentation remains available
- Progressive enhancement (works without JavaScript)
- Existing documentation URLs unchanged
- Sphinx workflow unchanged (enhancement via extensions)

## Alternatives Considered

### Alternative 1: Jupyter Notebooks Only

Distribute tutorials as downloadable notebooks.

**Pros:**
- Users familiar with Jupyter
- Full Python environment

**Cons:**
- Requires local installation
- Not discoverable via web search
- No parameter widgets

**Decision:** Complement, not replacement

### Alternative 2: Third-Party Platforms (Colab, Binder)

Link to external platforms for interactive examples.

**Pros:**
- No infrastructure to maintain
- Full Python environment

**Cons:**
- Requires external accounts
- Slower loading
- Inconsistent branding

**Decision:** Offer as alternative, but prefer in-browser

## Open Questions

1. **Performance with large models?**
   - Pyodide may struggle with large computations
   - Proposal: Offer cloud-backed execution for complex examples

2. **Versioning of interactive examples?**
   - Should examples match PyTorch version?
   - Proposal: Version-specific examples with fallback to latest

3. **Accessibility?**
   - Screen readers with interactive widgets
   - Keyboard navigation
   - Proposal: Full WCAG 2.1 AA compliance

## Success Metrics

1. **Engagement:** 50%+ increase in time-on-page for interactive docs
2. **Learning:** 30% reduction in "how do I..." forum questions
3. **Quality:** 95%+ example accuracy (automated testing)
4. **Adoption:** 10,000+ interactive example executions/day
5. **Search:** 70%+ users find answer in top 3 results

## References

- **Sphinx Configuration:** [`docs/source/conf.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/docs/source/conf.py)
- **JupyterLite:** https://jupyterlite.readthedocs.io/
- **Pyodide:** https://pyodide.org/
- **Related Work:** TensorFlow Playground, Observable notebooks, Rust Playground

---

**Next Steps:**
1. Prototype with JupyterLite on 5 pages
2. User testing with 20+ participants
3. Performance benchmarking
4. Refine based on feedback
5. Roll out to top 100 pages
