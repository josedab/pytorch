"""
Unit tests for SemanticDocSearch

Tests the semantic documentation search functionality.
"""

import unittest
from pathlib import Path
import tempfile
import json
from torch.testing._internal.common_utils import TestCase, run_tests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.semantic_search import SemanticDocSearch, SearchResult, FunctionInfo


class TestSemanticDocSearch(TestCase):
    """Test cases for SemanticDocSearch"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.doc_root = Path(self.temp_dir)
        self.index_path = Path(self.temp_dir) / 'index.json'

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test search initialization"""
        search = SemanticDocSearch()
        self.assertIsInstance(search.doc_index, list)
        self.assertIsInstance(search.type_signatures, dict)
        self.assertIsInstance(search.code_examples, dict)

    def test_save_and_load_index(self):
        """Test saving and loading search index"""
        search = SemanticDocSearch(self.index_path)

        # Add some test data
        search.doc_index = ['doc1', 'doc2']
        search.type_signatures = {
            'torch.tensor': FunctionInfo(
                name='torch.tensor',
                signature='(data) -> Tensor',
                input_types=['Any'],
                output_type='Tensor',
            )
        }
        search.code_examples = {
            'doc1': ['example1', 'example2']
        }

        # Save
        search._save_index()

        # Load in new instance
        search2 = SemanticDocSearch(self.index_path)

        self.assertEqual(search2.doc_index, ['doc1', 'doc2'])
        self.assertIn('torch.tensor', search2.type_signatures)
        self.assertEqual(search2.code_examples['doc1'], ['example1', 'example2'])

    def test_index_document(self):
        """Test indexing a single document"""
        # Create a test RST file
        doc_file = self.doc_root / 'test.rst'
        doc_file.write_text("""
Test Function
=============

.. code-block:: python

    import torch
    x = torch.tensor([1, 2, 3])
""")

        search = SemanticDocSearch()
        search._index_document(doc_file)

        # Check document was indexed
        self.assertIn(str(doc_file), search.doc_index)

        # Check code example was extracted
        self.assertIn(str(doc_file), search.code_examples)

    def test_search_basic(self):
        """Test basic keyword search"""
        # Create test documents
        doc1 = self.doc_root / 'tensor.rst'
        doc1.write_text("""
Tensor Operations
=================

Create a tensor with torch.tensor()
""")

        doc2 = self.doc_root / 'random.rst'
        doc2.write_text("""
Random Numbers
==============

Generate random tensors with torch.randn()
""")

        search = SemanticDocSearch()
        search.build_index(self.doc_root)

        # Search for "tensor"
        results = search.search('tensor', top_k=10)

        # Should find both documents
        self.assertGreater(len(results), 0)

    def test_search_by_signature(self):
        """Test search by type signature"""
        search = SemanticDocSearch()

        # Add some function signatures
        search.type_signatures = {
            'torch.add': FunctionInfo(
                name='torch.add',
                signature='(Tensor, Tensor) -> Tensor',
                input_types=['Tensor', 'Tensor'],
                output_type='Tensor',
            ),
            'torch.mul': FunctionInfo(
                name='torch.mul',
                signature='(Tensor, Tensor) -> Tensor',
                input_types=['Tensor', 'Tensor'],
                output_type='Tensor',
            ),
            'torch.randn': FunctionInfo(
                name='torch.randn',
                signature='(*size) -> Tensor',
                input_types=['int'],
                output_type='Tensor',
            ),
        }

        # Search for functions that take (Tensor, Tensor) -> Tensor
        results = search.search_by_signature(
            input_types=['Tensor', 'Tensor'],
            output_type='Tensor'
        )

        # Should find add and mul
        self.assertEqual(len(results), 2)
        names = [r.name for r in results]
        self.assertIn('torch.add', names)
        self.assertIn('torch.mul', names)

    def test_search_by_example(self):
        """Test search by code example"""
        search = SemanticDocSearch()

        # Add some code examples
        search.code_examples = {
            'doc1': ['import torch\nx = torch.randn(5)\ny = torch.relu(x)'],
            'doc2': ['import torch\nx = torch.tensor([1, 2, 3])'],
            'doc3': ['import numpy as np\nx = np.array([1, 2, 3])'],
        }
        search.doc_index = ['doc1', 'doc2', 'doc3']

        # Search for code similar to torch.randn example
        results = search.search_by_example('torch.randn(10)', top_k=5)

        # Should find doc1 (has randn)
        self.assertGreater(len(results), 0)

    def test_tokenize_code(self):
        """Test code tokenization"""
        search = SemanticDocSearch()

        code = """
import torch
x = torch.randn(5)
y = torch.relu(x)
"""

        tokens = search._tokenize_code(code)

        # Should extract key tokens
        self.assertIn('torch', tokens)
        self.assertIn('randn', tokens)
        self.assertIn('relu', tokens)

    def test_calculate_similarity(self):
        """Test similarity calculation"""
        search = SemanticDocSearch()

        tokens1 = {'torch', 'randn', 'relu'}
        tokens2 = {'torch', 'randn', 'sigmoid'}
        tokens3 = {'numpy', 'array', 'sum'}

        # Similar sets should have higher similarity
        sim1 = search._calculate_similarity(tokens1, tokens2)
        sim2 = search._calculate_similarity(tokens1, tokens3)

        self.assertGreater(sim1, sim2)

    def test_calculate_similarity_empty(self):
        """Test similarity with empty sets"""
        search = SemanticDocSearch()

        tokens1 = {'torch', 'randn'}
        tokens2 = set()

        sim = search._calculate_similarity(tokens1, tokens2)
        self.assertEqual(sim, 0.0)

    def test_extract_snippet(self):
        """Test snippet extraction"""
        search = SemanticDocSearch()

        content = """
This is a long document about PyTorch tensors.
Tensors are the fundamental data structure in PyTorch.
You can create tensors using torch.tensor().
There are many operations you can perform on tensors.
"""

        snippet = search._extract_snippet(content, 'tensor', max_length=100)

        # Should contain the query term
        self.assertIn('tensor', snippet.lower())

        # Should not exceed max length
        self.assertLessEqual(len(snippet), 110)  # Allow for '...'

    def test_path_to_url(self):
        """Test converting file path to URL"""
        search = SemanticDocSearch()

        path = '/path/to/docs/source/generated/torch.tensor.rst'
        url = search._path_to_url(path)

        # Should create valid URL
        self.assertIn('torch.tensor', url)
        self.assertTrue(url.endswith('.html'))

    def test_type_matches(self):
        """Test type matching"""
        search = SemanticDocSearch()

        # Exact match
        self.assertTrue(search._type_matches('Tensor', 'Tensor'))

        # Substring match
        self.assertTrue(search._type_matches('Tensor', 'torch.Tensor'))

        # Any type
        self.assertTrue(search._type_matches('Any', 'Tensor'))
        self.assertTrue(search._type_matches('Tensor', 'Any'))

        # No match
        self.assertFalse(search._type_matches('Tensor', 'int'))

    def test_types_match_list(self):
        """Test matching lists of types"""
        search = SemanticDocSearch()

        query_types = ['Tensor', 'Tensor']
        func_types1 = ['Tensor', 'Tensor', 'bool']
        func_types2 = ['Tensor', 'int']
        func_types3 = ['Tensor']

        # Should match if query types are prefix of function types
        self.assertTrue(search._types_match(query_types, func_types1))

        # Should not match if types don't align
        self.assertFalse(search._types_match(query_types, func_types2))

        # Should not match if function has fewer types
        self.assertFalse(search._types_match(query_types, func_types3))

    def test_build_index(self):
        """Test building complete search index"""
        # Create some test documents
        (self.doc_root / 'doc1.rst').write_text("""
Title
=====

Content here.

.. code-block:: python

    import torch
""")

        (self.doc_root / 'doc2.rst').write_text("""
Another Title
=============

More content.
""")

        search = SemanticDocSearch()
        search.build_index(self.doc_root)

        # Should have indexed documents
        self.assertEqual(len(search.doc_index), 2)


if __name__ == "__main__":
    run_tests()
