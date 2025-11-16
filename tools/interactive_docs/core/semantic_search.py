"""
Semantic Documentation Search

This module provides AI-powered semantic search for PyTorch documentation,
supporting text search, search by example, and search by type signature.

Based on RFC-0004: Interactive Documentation with Live Examples
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json


@dataclass
class SearchResult:
    """A search result"""
    title: str
    url: str
    score: float
    snippet: str
    api_name: Optional[str] = None
    category: Optional[str] = None


@dataclass
class FunctionInfo:
    """Information about a function"""
    name: str
    signature: str
    input_types: List[str]
    output_type: str
    docstring: Optional[str] = None


class SemanticDocSearch:
    """AI-powered documentation search"""

    def __init__(self, index_path: Optional[Path] = None):
        """
        Initialize the semantic search.

        Args:
            index_path: Path to pre-built search index (optional)
        """
        self.index_path = Path(index_path) if index_path else None
        self.doc_index: List[str] = []
        self.embeddings: Dict[str, Any] = {}
        self.type_signatures: Dict[str, FunctionInfo] = {}
        self.code_examples: Dict[str, List[str]] = {}

        if self.index_path and self.index_path.exists():
            self._load_index()

    def _load_index(self) -> None:
        """Load pre-built search index from disk"""
        if not self.index_path:
            return

        try:
            with open(self.index_path, 'r') as f:
                data = json.load(f)
                self.doc_index = data.get('doc_index', [])
                self.type_signatures = {
                    k: FunctionInfo(**v)
                    for k, v in data.get('type_signatures', {}).items()
                }
                self.code_examples = data.get('code_examples', {})
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    def _save_index(self) -> None:
        """Save search index to disk"""
        if not self.index_path:
            return

        data = {
            'doc_index': self.doc_index,
            'type_signatures': {
                k: {
                    'name': v.name,
                    'signature': v.signature,
                    'input_types': v.input_types,
                    'output_type': v.output_type,
                    'docstring': v.docstring,
                }
                for k, v in self.type_signatures.items()
            },
            'code_examples': self.code_examples,
        }

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def build_index(self, doc_root: Path) -> None:
        """
        Build search index from documentation.

        Args:
            doc_root: Root directory of documentation
        """
        self.doc_index = []
        self.type_signatures = {}
        self.code_examples = {}

        # Find all documentation files
        for doc_file in doc_root.glob('**/*.rst'):
            self._index_document(doc_file)

        # Build type signature index from PyTorch module
        self._index_type_signatures()

        # Save index
        self._save_index()

    def _index_document(self, doc_file: Path) -> None:
        """
        Index a single documentation file.

        Args:
            doc_file: Path to documentation file
        """
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract title
            title_match = re.search(r'^(.+)\n[=]+\n', content, re.MULTILINE)
            title = title_match.group(1) if title_match else doc_file.stem

            # Add to index
            self.doc_index.append(str(doc_file))

            # Extract code examples
            examples = re.findall(r'.. code-block:: python\n\n((?:    .*\n)*)', content)
            if examples:
                self.code_examples[str(doc_file)] = examples

        except Exception:
            pass

    def _index_type_signatures(self) -> None:
        """Build index of function type signatures"""
        try:
            import torch
            import inspect

            # Index common modules
            modules = [
                ('torch', torch),
                ('torch.nn.functional', torch.nn.functional),
            ]

            for module_name, module in modules:
                for name, obj in inspect.getmembers(module):
                    if name.startswith('_') or not callable(obj):
                        continue

                    try:
                        sig = inspect.signature(obj)
                        input_types = []
                        output_type = 'Any'

                        # Extract parameter types
                        for param in sig.parameters.values():
                            if param.annotation != inspect.Parameter.empty:
                                input_types.append(str(param.annotation))
                            else:
                                input_types.append('Any')

                        # Extract return type
                        if sig.return_annotation != inspect.Signature.empty:
                            output_type = str(sig.return_annotation)

                        full_name = f'{module_name}.{name}'
                        self.type_signatures[full_name] = FunctionInfo(
                            name=full_name,
                            signature=str(sig),
                            input_types=input_types,
                            output_type=output_type,
                            docstring=inspect.getdoc(obj),
                        )

                    except (ValueError, TypeError):
                        continue

        except ImportError:
            pass

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Semantic search for query.

        This is a simplified implementation. A full implementation would use
        embeddings and vector similarity search.

        Args:
            query: Search query
            top_k: Maximum number of results

        Returns:
            List of search results
        """
        results = []

        # Simple keyword-based search (would use embeddings in production)
        query_lower = query.lower()
        keywords = query_lower.split()

        for doc_path in self.doc_index:
            score = 0.0

            # Read document
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    content_lower = content.lower()

                    # Calculate simple relevance score
                    for keyword in keywords:
                        score += content_lower.count(keyword)

                    if score > 0:
                        # Extract snippet
                        snippet = self._extract_snippet(content, query_lower)

                        # Extract title
                        title_match = re.search(r'^(.+)\n[=]+\n', content, re.MULTILINE)
                        title = title_match.group(1) if title_match else Path(doc_path).stem

                        results.append(SearchResult(
                            title=title,
                            url=self._path_to_url(doc_path),
                            score=score,
                            snippet=snippet,
                        ))

            except Exception:
                continue

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def search_by_signature(self,
                          input_types: List[str],
                          output_type: Optional[str] = None) -> List[FunctionInfo]:
        """
        Search functions by type signature.

        Args:
            input_types: List of input type names (e.g., ['Tensor', 'Tensor'])
            output_type: Expected output type (e.g., 'Tensor')

        Returns:
            List of matching functions
        """
        matches = []

        for func_name, func_info in self.type_signatures.items():
            # Check if input types match
            input_match = self._types_match(input_types, func_info.input_types)

            # Check if output type matches
            output_match = (
                output_type is None or
                self._type_matches(output_type, func_info.output_type)
            )

            if input_match and output_match:
                matches.append(func_info)

        return matches

    def _types_match(self, query_types: List[str],
                    func_types: List[str]) -> bool:
        """
        Check if query types match function types.

        Args:
            query_types: Types from query
            func_types: Types from function signature

        Returns:
            True if types match
        """
        if len(query_types) > len(func_types):
            return False

        for query_type, func_type in zip(query_types, func_types):
            if not self._type_matches(query_type, func_type):
                return False

        return True

    def _type_matches(self, query_type: str, func_type: str) -> bool:
        """
        Check if a query type matches a function type.

        Args:
            query_type: Type from query
            func_type: Type from function

        Returns:
            True if types match
        """
        # Normalize types
        query_type = query_type.lower().strip()
        func_type = func_type.lower().strip()

        # Handle 'Any' type
        if query_type == 'any' or func_type == 'any':
            return True

        # Simple substring match
        return query_type in func_type or func_type in query_type

    def search_by_example(self, code: str, top_k: int = 10) -> List[SearchResult]:
        """
        Find similar code examples.

        Args:
            code: Code snippet to search for
            top_k: Maximum number of results

        Returns:
            List of search results with similar code
        """
        results = []

        # Extract key tokens from code
        code_tokens = self._tokenize_code(code)

        for doc_path, examples in self.code_examples.items():
            for example in examples:
                # Calculate similarity
                example_tokens = self._tokenize_code(example)
                similarity = self._calculate_similarity(code_tokens, example_tokens)

                if similarity > 0.1:  # Threshold
                    results.append(SearchResult(
                        title=Path(doc_path).stem,
                        url=self._path_to_url(doc_path),
                        score=similarity,
                        snippet=example[:200] + '...',
                    ))

        # Sort by similarity
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def _tokenize_code(self, code: str) -> set:
        """
        Extract tokens from code.

        Args:
            code: Code string

        Returns:
            Set of tokens
        """
        # Simple tokenization (would use AST in production)
        tokens = re.findall(r'\w+', code.lower())
        return set(tokens)

    def _calculate_similarity(self, tokens1: set, tokens2: set) -> float:
        """
        Calculate similarity between two token sets.

        Args:
            tokens1: First token set
            tokens2: Second token set

        Returns:
            Similarity score (0 to 1)
        """
        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def _extract_snippet(self, content: str, query: str,
                        max_length: int = 200) -> str:
        """
        Extract a relevant snippet from content.

        Args:
            content: Full content
            query: Search query
            max_length: Maximum snippet length

        Returns:
            Relevant snippet
        """
        # Find first occurrence of query
        query_pos = content.lower().find(query.lower())

        if query_pos == -1:
            # Return first paragraph
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip() and not para.startswith('..'):
                    snippet = para.strip()
                    if len(snippet) > max_length:
                        snippet = snippet[:max_length] + '...'
                    return snippet
            return content[:max_length] + '...'

        # Extract context around query
        start = max(0, query_pos - 50)
        end = min(len(content), query_pos + max_length)
        snippet = content[start:end].strip()

        if start > 0:
            snippet = '...' + snippet
        if end < len(content):
            snippet = snippet + '...'

        return snippet

    def _path_to_url(self, path: str) -> str:
        """
        Convert file path to documentation URL.

        Args:
            path: File path

        Returns:
            Documentation URL
        """
        # Convert path to URL
        path_obj = Path(path)
        rel_path = path_obj.stem
        return f'/docs/stable/{rel_path}.html'


class SearchIndex:
    """
    Build and manage search index.

    This would be expanded with actual embedding generation in production.
    """

    def __init__(self):
        self.search = SemanticDocSearch()

    def build(self, doc_root: Path, output_path: Path) -> None:
        """
        Build search index.

        Args:
            doc_root: Root directory of documentation
            output_path: Path to save index
        """
        self.search.build_index(doc_root)
        self.search.index_path = output_path
        self.search._save_index()

        print(f"Index built successfully:")
        print(f"  Documents: {len(self.search.doc_index)}")
        print(f"  Type signatures: {len(self.search.type_signatures)}")
        print(f"  Code examples: {len(self.search.code_examples)}")
        print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic search for PyTorch documentation"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Build index command
    build_parser = subparsers.add_parser('build', help='Build search index')
    build_parser.add_argument(
        '--doc-root',
        type=Path,
        required=True,
        help='Root directory of documentation'
    )
    build_parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for index'
    )

    # Search command
    search_parser = subparsers.add_parser('search', help='Search documentation')
    search_parser.add_argument(
        '--index',
        type=Path,
        required=True,
        help='Path to search index'
    )
    search_parser.add_argument(
        'query',
        help='Search query'
    )
    search_parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to return'
    )

    args = parser.parse_args()

    if args.command == 'build':
        index = SearchIndex()
        index.build(args.doc_root, args.output)

    elif args.command == 'search':
        search = SemanticDocSearch(args.index)
        results = search.search(args.query, args.top_k)

        print(f"Search results for: {args.query}")
        print("=" * 70)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Score: {result.score:.2f}")
            print(f"   {result.snippet}")
            print()
