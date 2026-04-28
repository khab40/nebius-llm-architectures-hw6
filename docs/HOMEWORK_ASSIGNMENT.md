# Homework Assignment: Build Your Own Tiny Transformer

## Goal

Implement a decoder-only Transformer language model from scratch in PyTorch and train it on character-level Tiny Shakespeare.

The homework notebook is:

```text
src/tiny_transformer_lm.ipynb
```

By the end, your model should generate text that looks roughly Shakespeare-like: speaker names, line breaks, punctuation, and fragments of English-like words.

## What You Must Implement

The notebook already provides the data pipeline, training loop, sanity checks, perplexity calculation, and sample generation. Your job is to complete the TODO sections in the model code.

Implement:

1. `MultiHeadSelfAttention`
2. `FeedForward`
3. `Block`
4. `TinyTransformerLM`
5. `TinyTransformerLM.generate`

Do not rewrite the whole notebook. Keep your changes focused on the TODO blocks unless you are only adjusting the runtime device for local Apple Silicon execution.

## Required Concepts

Your implementation should correctly use:

- token embeddings;
- positional embeddings;
- multi-head causal self-attention;
- scaled dot-product attention;
- a lower-triangular causal mask;
- dropout;
- residual connections;
- pre-norm LayerNorm;
- position-wise feed-forward networks;
- cross-entropy loss for next-character prediction;
- autoregressive token generation.

## Runtime Options

You may run the homework locally or in Nebius Cloud.

### Local Apple Silicon

Local VS Code execution is supported if PyTorch MPS is available.

The notebook currently uses:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

For Apple Silicon GPU execution, replace it with:

```python
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")
```

Run VS Code or Jupyter with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Nebius Cloud

Nebius Cloud is recommended for the fastest and most reproducible full training run. A T4-class CUDA GPU is enough for this homework.

The original device line should select CUDA automatically:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

More detailed run instructions are in:

```text
docs/RUNNING_DEMOS_AND_HOMEWORK.md
```

## VS Code Kernel Troubleshooting

If VS Code fails to start the notebook kernel, first make sure the project environment is synced:

```bash
uv sync
```

Then register the project environment as a named Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name hw6 --display-name "Python 3.14 (hw6)"
```

In VS Code, select:

```text
Select Kernel -> Jupyter Kernel -> Python 3.14 (hw6)
```

or:

```text
Select Kernel -> Python Environments -> .venv
```

If VS Code shows an extension-side error such as:

```text
Missing required @injectable annotation
```

reload VS Code and update or reinstall the `Jupyter` and `Python` extensions. If the issue still persists, bypass the VS Code Jupyter extension and run JupyterLab directly:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run jupyter lab
```

## Required Work

1. Run the setup and data cells.
2. Review the hyperparameters.
3. Implement `MultiHeadSelfAttention`.
4. Run the attention sanity check.
5. Implement `FeedForward`.
6. Run the feed-forward sanity check.
7. Implement `Block`.
8. Run the block sanity check.
9. Implement `TinyTransformerLM.forward`.
10. Implement `TinyTransformerLM.generate`.
11. Run the full-model sanity check.
12. Train the model.
13. Calculate validation perplexity and bits per character.
14. Generate final text samples.
15. Save the notebook with all cells executed.

## Questions To Answer

Include short answers in the notebook or in a short accompanying note:

1. Why does the attention layer need a causal mask?
2. What are queries, keys, and values used for in self-attention?
3. Why do we split attention into multiple heads?
4. Why are residual connections important in Transformer blocks?
5. What does perplexity measure for a language model?
6. Why must `generate` crop the context to the last `block_size` tokens?
7. What changed in the generated samples before and after training?

## Expected Artifacts

Submit the completed `src/tiny_transformer_lm.ipynb` notebook with:

- all TODO sections implemented;
- all sanity checks passing;
- training logs visible;
- final loss values visible;
- validation perplexity and bits per character visible;
- generated text samples visible.

## Success Criteria

Your work is complete when:

- no `raise NotImplementedError` lines remain in the TODO sections;
- all sanity-check cells pass;
- the model trains without runtime errors;
- final train loss is below `1.8`;
- validation loss is reasonably close to train loss;
- generated samples are mostly made of real-looking English words and punctuation;
- the notebook is saved with all cells executed.

## Practical Notes

The default notebook settings are sized to train in a few minutes on a Colab T4-style GPU. On Apple Silicon, MPS performance depends on the chip and available unified memory.

For quick local debugging, you may temporarily reduce:

```python
batch_size = 32
max_iters = 1000
eval_interval = 200
```

Use the original/default settings again for the final run if possible.
