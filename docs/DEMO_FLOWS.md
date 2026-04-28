# Demo Flows

This document describes the two main demo paths in `efficient_fine_tuning.ipynb`.

## Demo 1: Prompt Tuning

Prompt tuning freezes the base model and trains a small set of virtual prompt embeddings.

```mermaid
flowchart TD
    A[Start notebook] --> B[Install and import dependencies]
    B --> C[Load tokenizer and Pythia base model]
    C --> D[Load tweet_eval irony dataset]
    D --> E[Convert rows to generation format]
    E --> F[Generate baseline labels]
    F --> G[Create PromptTuningConfig]
    G --> H[Wrap model with PEFT prompt tuning adapter]
    H --> I[Train virtual prompt tokens]
    I --> J[Evaluate validation loss and perplexity]
    J --> K[Save adapter to models/prompt_tuning]
    K --> L[Reload base model plus prompt adapter]
    L --> M[Generate labels for test tweets]
    M --> N[Map generated text to irony or non irony]
    N --> O[Calculate prompt tuning accuracy]
```

## Demo 2: LoRA

LoRA freezes the base model and trains low-rank adapter matrices attached to selected transformer modules.

```mermaid
flowchart TD
    A[Restart runtime if GPU memory is low] --> B[Install and import dependencies]
    B --> C[Load tokenizer and Pythia base model]
    C --> D[Load tweet_eval irony dataset]
    D --> E[Convert rows to generation format]
    E --> F[Create LoraConfig]
    F --> G[Target attention and MLP modules]
    G --> H[Wrap model with PEFT LoRA adapter]
    H --> I[Train low-rank adapter weights]
    I --> J[Evaluate validation loss and perplexity]
    J --> K[Save adapter to models/lora]
    K --> L[Reload base model plus LoRA adapter]
    L --> M[Generate labels for test tweets]
    M --> N[Map generated text to irony or non irony]
    N --> O[Calculate LoRA accuracy]
```

## Comparison Flow

```mermaid
flowchart LR
    A[Prompt tuning results] --> C[Compare]
    B[LoRA results] --> C
    C --> D[Accuracy]
    C --> E[Trainable parameters]
    C --> F[GPU memory behavior]
    C --> G[Output format reliability]
    C --> H[Recommendation for the task]
```
