# MNIST Parallel Hyperparameter Tuning Demo

This document explains how to run:

```text
src/MNIST_Parallel_Hyperparameter_Tuning_Demo.ipynb
```

The notebook demonstrates a lightweight parallel hyperparameter search for an MNIST classifier. It creates multiple trial configs, launches worker processes, and shows a simple dashboard for status and results.

## What The Demo Does

The notebook has four main parts:

1. creates a working directory and downloads MNIST;
2. writes a worker script, `mnist_worker.py`, that trains one trial;
3. writes a launcher script, `mnist_launcher.py`, that keeps several trials running in parallel;
4. creates trial configs, starts the launcher, and displays a dashboard.

Each trial trains a small neural network with different hyperparameters such as learning rate, hidden size, dropout, batch size, and optimizer.

## Outputs

The demo writes generated files under a root directory:

```text
configs/
status/
results/
logs/
mnist_worker.py
mnist_launcher.py
```

In the original notebook this root is:

```python
ROOT = Path("/content/mnist_hpo_demo")
```

That path is Colab-specific. For local VS Code execution, use a project-local path instead.

## Run Locally In VS Code On Apple Silicon

### 1. Prepare the environment

From the project root:

```bash
uv sync
```

The project dependency set already includes `torchvision`.

### 2. Open the notebook

1. Open the project folder in VS Code.
2. Select the `.venv` Python interpreter.
3. Open `src/MNIST_Parallel_Hyperparameter_Tuning_Demo.ipynb`.
4. Select the `.venv` notebook kernel.

### 3. Change the root directory for local execution

Replace:

```python
ROOT = Path("/content/mnist_hpo_demo")
```

with:

```python
ROOT = Path("../mnist_hpo_demo").resolve()
```

when running from the `src/` notebook folder, or:

```python
ROOT = Path("mnist_hpo_demo").resolve()
```

if your notebook working directory is the project root.

The generated `mnist_hpo_demo/` directory is ignored by `.gitignore` through the generated-data/output patterns.

### 4. Apple Silicon GPU note

The worker script currently uses:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

On Apple Silicon this selects CPU, not GPU. For MPS execution, change it to:

```python
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

MNIST is small, so CPU execution is acceptable. MPS may help, but parallel worker contention can reduce the benefit.

For local Apple Silicon runs, start conservatively:

```python
make_trial_configs(n_trials=4, seed=123)
start_hpo(max_parallel=1)
```

Then increase `max_parallel` only if the machine stays responsive.

## Run In Nebius Cloud

Nebius Cloud is useful if you want CUDA behavior and faster parallel trials.

### 1. Prepare the environment

From the project root:

```bash
uv sync
```

The project dependency set already includes `torchvision`.

### 2. Verify CUDA

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected output should include:

```text
True
```

### 3. Run Jupyter

```bash
uv run jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
```

Open the printed Jupyter URL through the access method provided by your Nebius environment.

### 4. Run the demo

Open:

```text
src/MNIST_Parallel_Hyperparameter_Tuning_Demo.ipynb
```

On a CUDA GPU, the original device selection works:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Suggested starting point:

```python
make_trial_configs(n_trials=8, seed=123)
start_hpo(max_parallel=2)
```

If the GPU has enough memory and utilization is low, increase `max_parallel`.

## Dashboard

The dashboard cell can be rerun while trials are active. It reads the generated status and result files and reports progress.

Typical states:

- pending;
- running;
- completed;
- failed;
- stopped.

Use the stop helper in the notebook to ask the launcher to terminate active workers cleanly.

## Troubleshooting

If local execution fails because `/content/...` does not exist, replace the Colab path with a project-local `ROOT`.

If imports fail, confirm the notebook kernel is using `.venv`.

If `torchvision` import still fails, resync the environment:

```bash
uv sync
```

If the local machine becomes slow, reduce:

```python
max_parallel=1
n_trials=2
```

If CUDA memory is exhausted in Nebius Cloud, reduce `max_parallel` or lower the per-worker GPU memory fraction in the trial config.
