# Running Demos and Homework

This guide documents two ways to run the project notebooks:

1. locally in VS Code on Apple Silicon, using the MPS GPU backend where applicable;
2. in Nebius Cloud on a CUDA GPU.

The homework assignment is in:

```text
src/tiny_transformer_lm.ipynb
```

`src/efficient_fine_tuning.ipynb` is a PEFT demo/reference notebook, not the homework notebook.

## Notebook Recommendation

| Notebook | Purpose | Local Apple Silicon | Nebius Cloud |
| --- | --- | --- | --- |
| `src/tiny_transformer_lm.ipynb` | Homework: tiny GPT-style Transformer LM | Recommended | Recommended for fastest/reproducible training |
| `src/MNIST_Parallel_Hyperparameter_Tuning_Demo.ipynb` | Demo: parallel hyperparameter tuning | Recommended | Optional |
| `src/efficient_fine_tuning.ipynb` | Demo: PEFT prompt tuning and LoRA | Possible with edits | Recommended |

## Homework Scope

In `src/tiny_transformer_lm.ipynb`, students review a completed decoder-only Transformer implementation from scratch in PyTorch and train it on character-level Tiny Shakespeare.

The completed `# DONE` implementation blocks are:

- `MultiHeadSelfAttention`;
- `FeedForward`;
- `Block`;
- `TinyTransformerLM`;
- autoregressive `generate`.

The training loop, data loading, sanity checks, untrained baseline sample, timing summary, perplexity calculation, and sample generation are included.

Expected successful homework state:

- all `raise NotImplementedError` lines are removed;
- attention, feed-forward, block, and full-model sanity checks pass;
- an untrained baseline sample is visible before training;
- final train loss is below `1.8`;
- the timing summary reports end-to-end runtime, evaluation time, and train-only throughput;
- generated text looks roughly Shakespeare-like;
- all notebook cells are executed before submission.

## Option 1: Run Homework Locally in VS Code on Apple Silicon

### 1. Prepare the environment

From the project root:

```bash
uv sync
```

Verify that PyTorch sees the Apple Silicon GPU:

```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

Expected output:

```text
True
```

### 2. Open the notebook in VS Code

1. Open the project folder in VS Code.
2. Select the `.venv` Python interpreter.
3. Open `src/tiny_transformer_lm.ipynb`.
4. Select the `.venv` notebook kernel.

If launching VS Code from a terminal, use:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 code .
```

The fallback flag lets PyTorch run unsupported MPS operations on CPU instead of failing immediately.

### 3. Confirm Apple Silicon GPU selection

The notebook uses a portable device helper equivalent to:

```python
def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

On Apple Silicon, the setup cell should print `Using device: mps` when MPS is available.

### 4. Run the homework locally

Run the notebook in order:

1. setup;
2. data download and character vocabulary;
3. hyperparameters;
4. `# DONE` implementation blocks;
5. sanity checks;
6. untrained baseline sample;
7. training and timing summary;
8. perplexity;
9. trained sample generation.

The default notebook settings are sized for a Colab T4. On Apple Silicon, runtime depends heavily on the chip and memory size.

If local training is too slow, reduce these values temporarily for smoke tests:

```python
batch_size = 32
max_iters = 1000
eval_interval = 200
```

Use the original/default settings again for the final submitted run if possible.

### 5. Local troubleshooting

If the notebook runs on CPU instead of GPU, re-check the device cell and confirm:

```python
torch.backends.mps.is_available()
```

If MPS errors occur, start Jupyter/VS Code with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

If memory pressure is high, reduce `batch_size` first. Avoid changing the model architecture until the implementation blocks and sanity checks are understood.

## Option 2: Run Homework in Nebius Cloud on CUDA GPU

Nebius Cloud is the safest path for a full homework run because the notebook was sized for a T4-class CUDA GPU.

### 1. Start a CUDA GPU environment

Use a GPU instance with:

- an NVIDIA CUDA-capable GPU;
- at least T4-class performance;
- enough disk space for the Tiny Shakespeare data and Python environment.

The tiny transformer homework does not require a large LLM or huge VRAM. A modest CUDA GPU is enough.

### 2. Prepare the project

Copy or clone this project onto the Nebius instance, then enter the project root:

```bash
cd hw6
```

If `uv` is available:

```bash
uv sync
```

If `uv` is not available, use a normal virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch tqdm jupyterlab ipykernel ipywidgets
```

### 3. Verify CUDA

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected output should include:

```text
True
```

### 4. Start Jupyter

```bash
uv run jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
```

Open the printed Jupyter URL using the access method provided by your Nebius environment.

### 5. Run the homework notebook

Open:

```text
src/tiny_transformer_lm.ipynb
```

The portable device helper should select CUDA automatically:

```python
device = get_torch_device()
```

Recommended Nebius flow:

1. run setup and data cells;
2. review all `# DONE` implementation sections;
3. run each sanity-check cell;
4. generate the untrained baseline sample;
5. train with default hyperparameters;
6. review the timing summary and verify train loss goes below `1.8`;
7. calculate perplexity and bits per character;
8. generate trained text samples;
9. save the notebook with all cells executed.

## Running the Other Demo Notebooks

### `src/MNIST_Parallel_Hyperparameter_Tuning_Demo.ipynb`

Run locally first. It should be small enough for Apple Silicon, and it is useful for experimenting with local notebook execution.

If the notebook has CUDA-specific code, use the same portable selector:

```python
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

### `src/efficient_fine_tuning.ipynb`

This is a PEFT demo/reference notebook for prompt tuning and LoRA. It is not the homework assignment.

Use Nebius Cloud for the smooth path because the notebook is CUDA-oriented and uses `EleutherAI/pythia-1b-deduped`.

Local Apple Silicon execution is possible only after edits:

- replace hardcoded `.cuda()` calls with `.to(device)`;
- use the portable device selector;
- skip old notebook-local dependency pins if they conflict with the project environment;
- reduce batch size if memory pressure is high.

## Quick Decision Rule

Use local VS Code when:

- you are implementing and debugging `src/tiny_transformer_lm.ipynb`;
- you want quick sanity-check iterations;
- Apple Silicon MPS is available;
- training time is acceptable.

Use Nebius Cloud when:

- you want the fastest and most reproducible full homework training run;
- local MPS is slow or unstable;
- you are running the PEFT demo notebook;
- you need CUDA compatibility without editing notebook code.
