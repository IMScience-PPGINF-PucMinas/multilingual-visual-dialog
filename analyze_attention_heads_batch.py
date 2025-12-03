"""
Attention Head Analysis with Batch Processing
Optimized version that processes multiple examples simultaneously

Advantages:
- 5-10x faster than sequential processing
- Better GPU utilization
- Processes batches of 16-50 examples at once
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
from tqdm import tqdm
from collections import defaultdict

from vdbert_visual_dialog_loader import load_vdbert_complete


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "model.30.5.007.bin"

VISUAL_FEATURES = "data/img_feats1.0/train.h5"

DATASET_PT = "dataset/visdial_1.0_train-PT.json"
DATASET_ES = "dataset/visdial_1.0_train-ES.json"
DATASET_EN = "dataset/visdial_1.0_train-EN.json"

RESULTS_DIR = "results/vdbert_heads_batch"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# BATCH PROCESSING
# ============================================================================


def prepare_batch_inputs(texts, tokenizer, max_length=512):
    """
    Prepare batch of texts for the model

    Args:
        texts: List[str] - list of text inputs
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        batch_inputs: Dict with tensors for batch processing
    """
    num_vis_tokens = 36
    batch_size = len(texts)

    batch_input_ids = []
    batch_segment_ids = []
    batch_attention_mask = []

    max_seq_len = 0

    # Process each text
    for text in texts:
        text_encoding = tokenizer(text, add_special_tokens=False)
        text_token_ids = text_encoding["input_ids"][: max_length - num_vis_tokens - 3]

        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        unk_token_id = tokenizer.unk_token_id

        input_ids = (
            [cls_token_id]
            + [unk_token_id] * num_vis_tokens
            + [sep_token_id]
            + text_token_ids
            + [sep_token_id]
        )

        segment_ids = [0] * (num_vis_tokens + 2) + [1] * (len(text_token_ids) + 1)
        attention_mask = [1] * len(input_ids)

        batch_input_ids.append(input_ids)
        batch_segment_ids.append(segment_ids)
        batch_attention_mask.append(attention_mask)

        max_seq_len = max(max_seq_len, len(input_ids))

    # Pad to same length
    for i in range(batch_size):
        padding_length = max_seq_len - len(batch_input_ids[i])

        batch_input_ids[i] += [0] * padding_length
        batch_segment_ids[i] += [0] * padding_length
        batch_attention_mask[i] += [0] * padding_length

    # Convert to tensors
    batch_inputs = {
        "input_ids": torch.tensor(batch_input_ids),
        "token_type_ids": torch.tensor(batch_segment_ids),
        "attention_mask": torch.tensor(batch_attention_mask),
    }

    return batch_inputs


def get_batch_attention_patterns(model, tokenizer, texts, device="cpu"):
    """
    Extract attention patterns for BATCH of texts
    Processes ALL 12 layers simultaneously

    Args:
        model: VD-BERT model
        tokenizer: Tokenizer instance
        texts: List[str] - list of text inputs
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        all_attentions: List of arrays (batch_size, 12, num_heads, seq_len, seq_len)
                       Attention from ALL layers for each example
    """
    # Prepare batch
    batch_inputs = prepare_batch_inputs(texts, tokenizer)
    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

    # Forward pass with batch - returns attention from ALL layers
    with torch.no_grad():
        outputs = model(**batch_inputs, output_attentions=True)

    # outputs.attentions = tuple of 12 elements (one per layer)
    # Each element: (batch_size, num_heads, seq_len, seq_len)

    batch_size = len(texts)
    all_attentions = []

    # Reorganize: from (layers, batch) to (batch, layers)
    for batch_idx in range(batch_size):
        example_attentions = []
        for layer_idx in range(12):
            layer_attention = outputs.attentions[layer_idx][batch_idx]
            example_attentions.append(layer_attention.cpu().numpy())
        all_attentions.append(example_attentions)

    return all_attentions


def compute_head_activation_frequency_batch(
    loader, model, tokenizer, num_examples=100, batch_size=16, activation_threshold=0.1
):
    """
    Calculate activation frequency using BATCH PROCESSING
    MUCH FASTER than previous sequential version

    Args:
        loader: Data loader instance
        model: VD-BERT model
        tokenizer: Tokenizer instance
        num_examples: Total number of examples to analyze
        batch_size: Number of examples to process at once (16-32 recommended)
        activation_threshold: Threshold to consider a head activated

    Returns:
        activation_freq: Array of shape (num_layers, num_heads)
        activated_examples: Dict mapping (layer, head) to list of activated examples
    """
    num_layers = 12
    num_heads = 12

    activation_count = np.zeros((num_layers, num_heads))
    total_count = 0
    activated_examples = defaultdict(list)

    print(f"Analyzing {num_examples} examples in batches of {batch_size}...")

    # Collect texts in batches
    num_batches = (num_examples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_texts = []
        batch_metadata = []

        # Prepare batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_examples)

        for example_idx in range(start_idx, end_idx):
            try:
                dialog_idx = example_idx % len(loader.dialogs)
                turn_idx = np.random.randint(0, 10)

                example = loader.get_dialog_example(dialog_idx, turn_idx)
                text = example["question"]

                batch_texts.append(text)
                batch_metadata.append(
                    {
                        "example_idx": example_idx,
                        "dialog_idx": dialog_idx,
                        "turn_idx": turn_idx,
                        "text": text,
                    }
                )
            except:
                continue

        if not batch_texts:
            continue

        # Process entire batch at once
        batch_attentions = get_batch_attention_patterns(
            model, tokenizer, batch_texts, device=model.device
        )

        # Analyze each example in batch
        for example_attentions, metadata in zip(batch_attentions, batch_metadata):
            # For each layer
            for layer_idx in range(num_layers):
                layer_attention = example_attentions[layer_idx]

                # For each head
                for head_idx in range(num_heads):
                    head_attention = layer_attention[head_idx]
                    max_attention = head_attention.max()

                    # Activate if above threshold
                    if max_attention > activation_threshold:
                        activation_count[layer_idx, head_idx] += 1
                        activated_examples[(layer_idx, head_idx)].append(
                            {**metadata, "max_attention": float(max_attention)}
                        )

            total_count += 1

    # Normalize
    activation_freq = activation_count / max(total_count, 1)

    print(f"Processed {total_count} examples")

    return activation_freq, activated_examples


def compute_attention_statistics_batch(
    loader, model, tokenizer, num_examples=100, batch_size=16
):
    """
    Calculate detailed attention statistics in batch mode

    Args:
        loader: Data loader instance
        model: VD-BERT model
        tokenizer: Tokenizer instance
        num_examples: Total number of examples to analyze
        batch_size: Number of examples to process at once

    Returns:
        stats: Dict with statistics per layer and head
    """
    print(f"\nCalculating detailed statistics...")

    stats = {
        "mean_attention": np.zeros((12, 12)),  # Mean attention
        "max_attention": np.zeros((12, 12)),  # Maximum attention
        "std_attention": np.zeros((12, 12)),  # Standard deviation
        "activation_freq": np.zeros((12, 12)),  # Activation frequency
    }

    all_max_attentions = defaultdict(list)  # For calculating statistics

    num_batches = (num_examples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Computing statistics"):
        batch_texts = []

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_examples)

        for example_idx in range(start_idx, end_idx):
            try:
                dialog_idx = example_idx % len(loader.dialogs)
                turn_idx = np.random.randint(0, 10)
                example = loader.get_dialog_example(dialog_idx, turn_idx)
                batch_texts.append(example["question"])
            except:
                continue

        if not batch_texts:
            continue

        # Process batch
        batch_attentions = get_batch_attention_patterns(
            model, tokenizer, batch_texts, device=model.device
        )

        # Collect statistics
        for example_attentions in batch_attentions:
            for layer_idx in range(12):
                layer_attention = example_attentions[layer_idx]

                for head_idx in range(12):
                    head_attention = layer_attention[head_idx]
                    max_att = head_attention.max()

                    all_max_attentions[(layer_idx, head_idx)].append(max_att)

    # Calculate aggregated statistics
    for layer_idx in range(12):
        for head_idx in range(12):
            attentions = all_max_attentions[(layer_idx, head_idx)]

            if attentions:
                stats["mean_attention"][layer_idx, head_idx] = np.mean(attentions)
                stats["max_attention"][layer_idx, head_idx] = np.max(attentions)
                stats["std_attention"][layer_idx, head_idx] = np.std(attentions)
                stats["activation_freq"][layer_idx, head_idx] = np.mean(
                    [a > 0.1 for a in attentions]
                )

    return stats


# ============================================================================
# IMPROVED VISUALIZATION
# ============================================================================


def plot_comprehensive_analysis(
    stats_A, stats_B, label_A="Condition A", label_B="Condition B", save_path=None
):
    """
    COMPREHENSIVE visualization with multiple metrics
    Shows ALL 12 layers on Y-axis

    Args:
        stats_A: Statistics dict for condition A
        stats_B: Statistics dict for condition B
        label_A: Label for condition A
        label_B: Label for condition B
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    cmap = plt.cm.YlOrRd

    metrics = [
        ("activation_freq", "Activation Frequency"),
        ("mean_attention", "Mean Attention"),
        ("std_attention", "Std Attention"),
    ]

    for col, (metric, title) in enumerate(metrics):
        # Condition A
        ax_a = axes[0, col]
        im_a = ax_a.imshow(
            stats_A[metric],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=stats_A[metric].max(),
            origin="lower",  # Layer 0 at bottom, 11 at top
        )
        ax_a.set_title(f"{label_A}: {title}", fontsize=14, weight="bold")
        ax_a.set_xlabel("Attention Head", fontsize=12)
        ax_a.set_ylabel("Layer", fontsize=12)

        # Ensure all layers appear
        ax_a.set_yticks(range(12))
        ax_a.set_yticklabels(range(12))
        ax_a.set_xticks(range(12))
        ax_a.set_xticklabels(range(12))

        plt.colorbar(im_a, ax=ax_a, fraction=0.046)

        # Condition B
        ax_b = axes[1, col]
        im_b = ax_b.imshow(
            stats_B[metric],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=stats_B[metric].max(),
            origin="lower",  # Layer 0 at bottom, 11 at top
        )
        ax_b.set_title(f"{label_B}: {title}", fontsize=14, weight="bold")
        ax_b.set_xlabel("Attention Head", fontsize=12)
        ax_b.set_ylabel("Layer", fontsize=12)

        # Ensure all layers appear
        ax_b.set_yticks(range(12))
        ax_b.set_yticklabels(range(12))
        ax_b.set_xticks(range(12))
        ax_b.set_xticklabels(range(12))

        plt.colorbar(im_b, ax=ax_b, fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comprehensive analysis saved: {save_path}")

    plt.show()


def plot_paper_style_comparison(
    freq_A, freq_B, label_A="Condition A", label_B="Condition B", save_path=None
):
    """
    Visualization IDENTICAL to paper style
    Two heatmaps side by side with ALL 12 layers

    Args:
        freq_A: Activation frequency for condition A
        freq_B: Activation frequency for condition B
        label_A: Label for condition A
        label_B: Label for condition B
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    cmap_heatmap = plt.cm.Greys

    # Panel 1: Condition A
    ax1 = axes[0]
    im1 = ax1.imshow(
        freq_A,
        aspect="auto",
        cmap=cmap_heatmap,
        vmin=0,
        vmax=1,
        origin="lower",  # Layer 0 at bottom
        extent=[0, 12, 0, 12],  # x: 0-12 (heads), y: 0-12 (layers)
    )

    ax1.set_xlabel("Attention Head Indices", fontsize=13, weight="bold")
    ax1.set_ylabel("Layer", fontsize=13, weight="bold")
    ax1.set_title(f"{label_A}", fontsize=15, weight="bold", pad=15)

    # Ticks for ALL layers and heads
    ax1.set_xticks(np.arange(0, 12) + 0.5)
    ax1.set_xticklabels(range(12), fontsize=11)
    ax1.set_yticks(np.arange(0, 12) + 0.5)
    ax1.set_yticklabels(range(12), fontsize=11)

    # Grid
    ax1.set_xticks(np.arange(0, 13), minor=True)
    ax1.set_yticks(np.arange(0, 13), minor=True)
    ax1.grid(which="minor", color="white", linestyle="-", linewidth=1)

    # Panel 2: Condition B
    ax2 = axes[1]
    im2 = ax2.imshow(
        freq_B,
        aspect="auto",
        cmap=cmap_heatmap,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=[0, 12, 0, 12],
    )

    ax2.set_xlabel("Attention Head Indices", fontsize=13, weight="bold")
    ax2.set_ylabel("Layer", fontsize=13, weight="bold")
    ax2.set_title(f"{label_B}", fontsize=15, weight="bold", pad=15)

    ax2.set_xticks(np.arange(0, 12) + 0.5)
    ax2.set_xticklabels(range(12), fontsize=11)
    ax2.set_yticks(np.arange(0, 12) + 0.5)
    ax2.set_yticklabels(range(12), fontsize=11)

    ax2.set_xticks(np.arange(0, 13), minor=True)
    ax2.set_yticks(np.arange(0, 13), minor=True)
    ax2.grid(which="minor", color="white", linestyle="-", linewidth=1)

    # Shared colorbar
    cbar = fig.colorbar(im2, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("Activation Frequency", fontsize=13, weight="bold")

    # Add significant differences
    diff = freq_A - freq_B
    threshold = 0.2

    for ax, is_A in [(ax1, True), (ax2, False)]:
        for layer in range(12):
            for head in range(12):
                d = diff[layer, head]

                # Mark specialized heads
                if is_A and d > threshold:
                    # Specific to A
                    ax.plot(
                        head + 0.5,
                        layer + 0.5,
                        "o",
                        color="cyan",
                        markersize=10,
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                    )
                elif not is_A and d < -threshold:
                    # Specific to B
                    ax.plot(
                        head + 0.5,
                        layer + 0.5,
                        "o",
                        color="orange",
                        markersize=10,
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                    )
                elif abs(d) < 0.1 and freq_A[layer, head] > 0.3:
                    # Shared
                    ax.plot(
                        head + 0.5,
                        layer + 0.5,
                        "s",
                        color="lightgreen",
                        markersize=7,
                        markeredgecolor="black",
                        markeredgewidth=1,
                    )

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="cyan",
            markersize=10,
            markeredgecolor="black",
            label=f"{label_A} Specific",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markersize=10,
            markeredgecolor="black",
            label=f"{label_B} Specific",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="lightgreen",
            markersize=8,
            markeredgecolor="black",
            label="Shared",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Paper-style visualization saved: {save_path}")

    plt.show()


# ============================================================================
# OPTIMIZED EXPERIMENTS
# ============================================================================


def experiment_pt_vs_es_batch(num_examples=100, batch_size=16):
    """
    PT vs ES experiment with BATCH PROCESSING

    Args:
        num_examples: Number of examples to analyze
        batch_size: Batch size for processing
    """
    print("=" * 70)
    print("BATCH EXPERIMENT: PORTUGUESE vs SPANISH")
    print("=" * 70)
    print(f"\nProcessing {num_examples} examples in batches of {batch_size}")
    print()

    # Load PT
    print("[1/5] Loading model and PT dataset...")
    loader_pt, model, tokenizer_pt, _ = load_vdbert_complete(
        MODEL_PATH, VISUAL_FEATURES, DATASET_PT
    )

    # Analyze PT (BATCH)
    print("\n[2/5] Analyzing PORTUGUESE (batch processing)...")
    freq_pt, examples_pt = compute_head_activation_frequency_batch(
        loader_pt, model, tokenizer_pt, num_examples=num_examples, batch_size=batch_size
    )

    # Detailed statistics PT
    print("\n[3/5] Calculating PT statistics...")
    stats_pt = compute_attention_statistics_batch(
        loader_pt, model, tokenizer_pt, num_examples=num_examples, batch_size=batch_size
    )

    # Load ES
    print("\n[4/5] Loading ES dataset...")
    from transformers import BertTokenizer
    import json

    tokenizer_es = BertTokenizer.from_pretrained(
        "dccuchile/bert-base-spanish-wwm-cased"
    )

    with open(DATASET_ES, "r") as f:
        data_es = json.load(f)

    class SimpleLoader:
        def __init__(self, data):
            self.dialogs = data["data"]["dialogs"]
            self.questions = data["data"]["questions"]
            self.answers = data["data"]["answers"]

        def get_dialog_example(self, dialog_idx, turn_idx):
            dialog = self.dialogs[dialog_idx]
            turn = dialog["dialog"][turn_idx]
            return {
                "question": self.questions[turn["question"]],
                "answer": self.answers[turn["answer"]],
            }

    loader_es = SimpleLoader(data_es)

    # Analyze ES (BATCH)
    print("Analyzing SPANISH (batch processing)...")
    freq_es, examples_es = compute_head_activation_frequency_batch(
        loader_es, model, tokenizer_es, num_examples=num_examples, batch_size=batch_size
    )

    # ES statistics
    print("\n[5/5] Calculating ES statistics...")
    stats_es = compute_attention_statistics_batch(
        loader_es, model, tokenizer_es, num_examples=num_examples, batch_size=batch_size
    )

    # Comparative analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    diff = freq_pt - freq_es

    # Most different heads
    most_pt_specific = np.unravel_index(diff.argmax(), diff.shape)
    most_es_specific = np.unravel_index(diff.argmin(), diff.shape)

    print(f"\nMost PT-specific head:")
    print(f"  Layer {most_pt_specific[0]}, Head {most_pt_specific[1]}")
    print(f"  Freq PT: {freq_pt[most_pt_specific]:.3f}")
    print(f"  Freq ES: {freq_es[most_pt_specific]:.3f}")
    print(f"  Diff: {diff[most_pt_specific]:.3f}")

    print(f"\nMost ES-specific head:")
    print(f"  Layer {most_es_specific[0]}, Head {most_es_specific[1]}")
    print(f"  Freq PT: {freq_pt[most_es_specific]:.3f}")
    print(f"  Freq ES: {freq_es[most_es_specific]:.3f}")
    print(f"  Diff: {diff[most_es_specific]:.3f}")

    # Complete visualization
    print("\nCreating visualizations...")

    # Visualization 1: Paper style (ALL layers visible)
    plot_paper_style_comparison(
        freq_pt,
        freq_es,
        label_A="Portuguese",
        label_B="Spanish",
        save_path=os.path.join(RESULTS_DIR, "batch_PT_vs_ES_paper_style.png"),
    )

    # Visualization 2: Detailed analysis
    plot_comprehensive_analysis(
        stats_pt,
        stats_es,
        label_A="Portuguese",
        label_B="Spanish",
        save_path=os.path.join(RESULTS_DIR, "batch_PT_vs_ES_comprehensive.png"),
    )

    # Save data
    np.savez(
        os.path.join(RESULTS_DIR, "batch_PT_vs_ES_data.npz"),
        freq_pt=freq_pt,
        freq_es=freq_es,
        diff=diff,
        **{f"stats_pt_{k}": v for k, v in stats_pt.items()},
        **{f"stats_es_{k}": v for k, v in stats_es.items()},
    )
    print(f"Data saved: batch_PT_vs_ES_data.npz")

    return stats_pt, stats_es


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main execution function"""
    print("=" * 70)
    print("BATCH ATTENTION HEAD ANALYSIS")
    print("=" * 70)
    print("\nOPTIMIZED version with batch processing")
    print("5-10x faster than sequential version!\n")

    print("Configuration:")
    num_examples = 100
    batch_size = int(input("Batch size (8-32): ") or "16")

    print(f"\nAnalyzing {num_examples} examples")
    print(f"Batch size: {batch_size}")
    print(f"Estimated time: ~{(num_examples/batch_size)*0.5:.1f} minutes\n")

    experiment_pt_vs_es_batch(num_examples=num_examples, batch_size=batch_size)

    print("\n" + "=" * 70)
    print("BATCH ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nResults in: {RESULTS_DIR}/")
    print("  - batch_PT_vs_ES_paper_style.png")
    print("    --> Paper-style visualization (ALL 12 layers)")
    print("  - batch_PT_vs_ES_comprehensive.png")
    print("    --> Detailed analysis with multiple metrics")
    print("  - batch_PT_vs_ES_data.npz")
    print("    --> Raw data for further analysis")


if __name__ == "__main__":
    main()
