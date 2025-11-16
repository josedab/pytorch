/**
 * Interactive Documentation JavaScript
 *
 * Provides interactive code execution using Pyodide (Python in WebAssembly)
 * and parameter exploration widgets for PyTorch documentation.
 *
 * Based on RFC-0004: Interactive Documentation with Live Examples
 */

// Global Pyodide instance
let pyodide = null;
let pyodideReady = false;

// Monaco editor instances
const editors = new Map();

// Original code for reset functionality
const originalCode = new Map();

/**
 * Initialize Pyodide (Python in WebAssembly)
 */
async function initializePyodide() {
    try {
        console.log('Loading Pyodide...');
        const statusElements = document.querySelectorAll('.example-status');
        statusElements.forEach(el => {
            el.textContent = 'Loading Python environment...';
            el.className = 'example-status loading';
        });

        // Load Pyodide
        pyodide = await loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/'
        });

        // Install PyTorch (if available in Pyodide)
        // Note: Full PyTorch may not be available, use lightweight version
        try {
            await pyodide.loadPackage(['numpy', 'matplotlib']);
            console.log('Loaded NumPy and Matplotlib');
        } catch (e) {
            console.warn('Could not load some packages:', e);
        }

        pyodideReady = true;
        console.log('Pyodide ready!');

        statusElements.forEach(el => {
            el.textContent = 'Ready';
            el.className = 'example-status ready';
        });

    } catch (error) {
        console.error('Failed to initialize Pyodide:', error);
        const statusElements = document.querySelectorAll('.example-status');
        statusElements.forEach(el => {
            el.textContent = 'Failed to load Python environment';
            el.className = 'example-status error';
        });
    }
}

/**
 * Initialize Monaco code editors
 */
function initializeEditors() {
    // Find all code editor containers
    const editorContainers = document.querySelectorAll('[id$="-editor"]');

    editorContainers.forEach(container => {
        const exampleId = container.id.replace('-editor', '');
        const code = container.textContent.trim();

        // Store original code
        originalCode.set(exampleId, code);

        // Create Monaco editor
        require.config({
            paths: {
                'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs'
            }
        });

        require(['vs/editor/editor.main'], function() {
            const editor = monaco.editor.create(container, {
                value: code,
                language: 'python',
                theme: 'vs-light',
                minimap: { enabled: false },
                lineNumbers: 'on',
                scrollBeyondLastLine: false,
                automaticLayout: true,
                fontSize: 14,
            });

            editors.set(exampleId, editor);
        });
    });
}

/**
 * Run a code example
 */
async function runExample(exampleId) {
    if (!pyodideReady) {
        updateStatus(exampleId, 'Python environment not ready yet...', 'loading');
        return;
    }

    const editor = editors.get(exampleId);
    if (!editor) {
        console.error('Editor not found:', exampleId);
        return;
    }

    const code = editor.getValue();
    const outputContainer = document.querySelector(`#${exampleId}-output .output-content`);

    if (!outputContainer) {
        console.error('Output container not found:', exampleId);
        return;
    }

    // Clear previous output
    outputContainer.innerHTML = '';

    // Update status
    updateStatus(exampleId, 'Running...', 'running');

    try {
        // Capture stdout
        let output = '';
        pyodide.setStdout({
            batched: (text) => {
                output += text + '\n';
            }
        });

        // Run the code
        const startTime = performance.now();
        const result = await pyodide.runPythonAsync(code);
        const executionTime = (performance.now() - startTime).toFixed(2);

        // Display output
        if (output) {
            const pre = document.createElement('pre');
            pre.textContent = output;
            outputContainer.appendChild(pre);
        }

        // Display result if any
        if (result !== undefined && result !== null) {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result';
            resultDiv.textContent = `Result: ${result}`;
            outputContainer.appendChild(resultDiv);
        }

        // Check for matplotlib figures
        const figures = pyodide.globals.get('plt');
        if (figures && typeof figures.show === 'function') {
            // Matplotlib integration would go here
            // This is simplified - full implementation would capture plot data
            const plotDiv = document.createElement('div');
            plotDiv.textContent = '[Plot output would appear here]';
            plotDiv.className = 'plot-placeholder';
            outputContainer.appendChild(plotDiv);
        }

        updateStatus(exampleId, `Completed in ${executionTime}ms`, 'success');

    } catch (error) {
        // Display error
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-output';
        errorDiv.textContent = error.message;
        outputContainer.appendChild(errorDiv);

        updateStatus(exampleId, 'Error', 'error');
        console.error('Execution error:', error);
    }

    // Show output container
    const outputParent = outputContainer.parentElement;
    if (outputParent) {
        outputParent.style.display = 'block';
    }
}

/**
 * Reset an example to its original code
 */
function resetExample(exampleId) {
    const editor = editors.get(exampleId);
    if (!editor) {
        console.error('Editor not found:', exampleId);
        return;
    }

    const originalCodeValue = originalCode.get(exampleId);
    if (originalCodeValue) {
        editor.setValue(originalCodeValue);
    }

    // Clear output
    const outputContainer = document.querySelector(`#${exampleId}-output .output-content`);
    if (outputContainer) {
        outputContainer.innerHTML = '';
    }

    updateStatus(exampleId, 'Ready', 'ready');
}

/**
 * Share an example (copy link to clipboard)
 */
function shareExample(exampleId) {
    const editor = editors.get(exampleId);
    if (!editor) {
        return;
    }

    const code = editor.getValue();

    // Encode code in URL
    const encoded = btoa(encodeURIComponent(code));
    const url = `${window.location.origin}${window.location.pathname}?example=${exampleId}&code=${encoded}`;

    // Copy to clipboard
    navigator.clipboard.writeText(url).then(() => {
        updateStatus(exampleId, 'Link copied to clipboard!', 'success');
        setTimeout(() => {
            updateStatus(exampleId, 'Ready', 'ready');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        updateStatus(exampleId, 'Failed to copy link', 'error');
    });
}

/**
 * Update example status
 */
function updateStatus(exampleId, message, statusClass) {
    const statusElement = document.getElementById(`${exampleId}-status`);
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `example-status ${statusClass}`;
    }
}

/**
 * Load code from URL parameters
 */
function loadCodeFromURL() {
    const params = new URLSearchParams(window.location.search);
    const exampleId = params.get('example');
    const encodedCode = params.get('code');

    if (exampleId && encodedCode) {
        try {
            const code = decodeURIComponent(atob(encodedCode));
            const editor = editors.get(exampleId);
            if (editor) {
                editor.setValue(code);
            }
        } catch (e) {
            console.error('Failed to load code from URL:', e);
        }
    }
}

/**
 * Update parameter explorer example
 */
function updateParameterExample() {
    const sizeSlider = document.getElementById('size-slider');
    const distributionSelect = document.getElementById('distribution-select');

    if (!sizeSlider || !distributionSelect) {
        return;
    }

    const size = parseInt(sizeSlider.value);
    const distribution = distributionSelect.value;

    // Generate code based on parameters
    let code = 'import torch\nimport matplotlib.pyplot as plt\n\n';

    switch (distribution) {
        case 'normal':
            code += `x = torch.randn(${size})\n`;
            break;
        case 'uniform':
            code += `x = torch.rand(${size})\n`;
            break;
        case 'zeros':
            code += `x = torch.zeros(${size})\n`;
            break;
    }

    code += `
y = torch.relu(x)

# Calculate statistics
print(f"Input - Mean: {x.mean():.4f}, Std: {x.std():.4f}")
print(f"Output - Mean: {y.mean():.4f}, Std: {y.std():.4f}")
print(f"Zeros in output: {(y == 0).sum().item()} / {y.numel()}")

# Visualize
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(x.numpy(), bins=30, alpha=0.7, label='Input')
plt.hist(y.numpy(), bins=30, alpha=0.7, label='Output')
plt.legend()
plt.title('Distribution')

plt.subplot(1, 2, 2)
plt.scatter(range(min(100, ${size})), x[:100].numpy(), alpha=0.5, label='Input')
plt.scatter(range(min(100, ${size})), y[:100].numpy(), alpha=0.5, label='Output')
plt.legend()
plt.title('Sample Values')
plt.tight_layout()
plt.show()
`;

    // Run the code
    runParameterExplorerCode(code);
}

/**
 * Run code in parameter explorer
 */
async function runParameterExplorerCode(code) {
    if (!pyodideReady) {
        return;
    }

    const outputContainer = document.getElementById('param-explorer-output');
    if (!outputContainer) {
        return;
    }

    try {
        await pyodide.runPythonAsync(code);
    } catch (error) {
        console.error('Parameter explorer error:', error);
    }
}

/**
 * Initialize size slider display
 */
document.addEventListener('DOMContentLoaded', function() {
    const sizeSlider = document.getElementById('size-slider');
    const sizeValue = document.getElementById('size-value');

    if (sizeSlider && sizeValue) {
        sizeSlider.addEventListener('input', function() {
            sizeValue.textContent = this.value;
        });
    }

    // Load code from URL if present
    setTimeout(loadCodeFromURL, 100);
});

// Export functions for global access
window.runExample = runExample;
window.resetExample = resetExample;
window.shareExample = shareExample;
window.updateParameterExample = updateParameterExample;
window.initializePyodide = initializePyodide;
window.initializeEditors = initializeEditors;
