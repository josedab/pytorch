"""
Sphinx Extension for Interactive Examples

This Sphinx extension adds support for interactive code examples
that can be executed in the browser using JupyterLite/Pyodide.

Based on RFC-0004: Interactive Documentation with Live Examples

Usage in RST:

.. interactive-example::
   :id: example-relu

   import torch
   x = torch.randn(5)
   y = torch.relu(x)
   print(y)

"""

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from typing import List, Any
import hashlib


class InteractiveExampleNode(nodes.General, nodes.Element):
    """Node for interactive code examples"""
    pass


class InteractiveExample(SphinxDirective):
    """
    Directive for creating interactive code examples.

    Options:
        :id: Unique identifier for the example
        :language: Programming language (default: python)
        :hide-code: Hide the code initially (default: False)
        :hide-output: Hide the output initially (default: False)
    """

    has_content = True
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        'id': directives.unchanged,
        'language': directives.unchanged,
        'hide-code': directives.flag,
        'hide-output': directives.flag,
    }

    def run(self) -> List[nodes.Node]:
        """Process the directive"""
        # Get options
        example_id = self.options.get('id', self._generate_id())
        language = self.options.get('language', 'python')
        hide_code = 'hide-code' in self.options
        hide_output = 'hide-output' in self.options

        # Get code content
        code = '\n'.join(self.content)

        # Create node
        node = InteractiveExampleNode()
        node['code'] = code
        node['id'] = example_id
        node['language'] = language
        node['hide_code'] = hide_code
        node['hide_output'] = hide_output

        return [node]

    def _generate_id(self) -> str:
        """Generate a unique ID for the example"""
        code = '\n'.join(self.content)
        hash_obj = hashlib.md5(code.encode())
        return f"example-{hash_obj.hexdigest()[:8]}"


class ParameterWidget(SphinxDirective):
    """
    Directive for parameter exploration widgets.

    Usage:
        .. param-widget:: dropout_prob
           :type: slider
           :min: 0
           :max: 1
           :step: 0.1
           :default: 0.5
    """

    has_content = False
    required_arguments = 1  # parameter name
    optional_arguments = 0
    option_spec = {
        'type': directives.unchanged,  # slider, dropdown, checkbox
        'min': directives.unchanged,
        'max': directives.unchanged,
        'step': directives.unchanged,
        'default': directives.unchanged,
        'options': directives.unchanged,  # for dropdown
    }

    def run(self) -> List[nodes.Node]:
        """Process the directive"""
        param_name = self.arguments[0]
        widget_type = self.options.get('type', 'slider')

        node = nodes.container()
        node['classes'].append('param-widget')
        node['param_name'] = param_name
        node['widget_type'] = widget_type

        for key in ['min', 'max', 'step', 'default', 'options']:
            if key in self.options:
                node[key] = self.options[key]

        return [node]


class TensorViewer(SphinxDirective):
    """
    Directive for tensor visualization.

    Usage:
        .. tensor-viewer::
           :tensor: output
           :mode: image
           :colormap: viridis
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        'tensor': directives.unchanged,
        'mode': directives.unchanged,  # image, filters, 3d, table
        'colormap': directives.unchanged,
        'channel': directives.unchanged,
    }

    def run(self) -> List[nodes.Node]:
        """Process the directive"""
        node = nodes.container()
        node['classes'].append('tensor-viewer')

        for key in ['tensor', 'mode', 'colormap', 'channel']:
            if key in self.options:
                node[key] = self.options[key]

        return [node]


def visit_interactive_example_node_html(
    self: Any,
    node: InteractiveExampleNode
) -> None:
    """Render interactive example as HTML"""
    code = node['code']
    example_id = node['id']
    language = node['language']
    hide_code = node.get('hide_code', False)
    hide_output = node.get('hide_output', False)

    # Generate HTML
    html = f'''
<div class="interactive-example" id="{example_id}">
    <div class="example-header">
        <button class="run-button" onclick="runExample('{example_id}')">
            ▶ Run
        </button>
        <button class="reset-button" onclick="resetExample('{example_id}')">
            ↻ Reset
        </button>
        <span class="example-status" id="{example_id}-status"></span>
    </div>
    <div class="code-editor" {'style="display:none;"' if hide_code else ''}>
        <pre><code class="language-{language}" id="{example_id}-code">{code}</code></pre>
    </div>
    <div class="code-output" id="{example_id}-output" {'style="display:none;"' if hide_output else ''}></div>
</div>
'''

    self.body.append(html)


def depart_interactive_example_node_html(
    self: Any,
    node: InteractiveExampleNode
) -> None:
    """Close interactive example HTML"""
    pass


def visit_interactive_example_node_text(
    self: Any,
    node: InteractiveExampleNode
) -> None:
    """Render interactive example as plain text"""
    code = node['code']
    self.add_text(f"Interactive Example:\n\n{code}\n\n")


def depart_interactive_example_node_text(
    self: Any,
    node: InteractiveExampleNode
) -> None:
    """Close interactive example text"""
    pass


def add_static_files(app: Sphinx, config: Any) -> None:
    """Add CSS and JS files to the documentation"""
    # Add CSS
    app.add_css_file('interactive-docs.css')

    # Add JavaScript
    app.add_js_file('pyodide.js')  # Pyodide for Python execution
    app.add_js_file('interactive-docs.js')  # Our custom JS


def setup(app: Sphinx) -> dict:
    """
    Setup the Sphinx extension.

    Args:
        app: Sphinx application instance

    Returns:
        Extension metadata
    """
    # Add configuration values
    app.add_config_value('interactive_docs_enabled', True, 'html')
    app.add_config_value('pyodide_url', 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/', 'html')

    # Add directives
    app.add_directive('interactive-example', InteractiveExample)
    app.add_directive('param-widget', ParameterWidget)
    app.add_directive('tensor-viewer', TensorViewer)

    # Add node
    app.add_node(
        InteractiveExampleNode,
        html=(visit_interactive_example_node_html, depart_interactive_example_node_html),
        text=(visit_interactive_example_node_text, depart_interactive_example_node_text),
        latex=(visit_interactive_example_node_text, depart_interactive_example_node_text),
    )

    # Connect events
    app.connect('config-inited', add_static_files)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
