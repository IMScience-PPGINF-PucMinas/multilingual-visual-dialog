"""
VD-BERT Model Comparison: Cross-Modal Attention Analysis
Compares attention patterns across all layers between two different VD-BERT models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
from tqdm import tqdm

from vdbert_visual_dialog_loader import load_vdbert_complete


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH_PT = "initial_models/model.29.5.029.bin"
MODEL_PATH_MULT = "initial_models/bert_es/model.20.5.091.bin"

DATASET_PATH_PT = "dataset/visdial_1.0_val-PT.json"
DATASET_PATH_ES = "dataset/visdial_1.0_val-ES.json"

VISUAL_FEATURES = "data/img_feats1.0/train.h5"

RESULTS_DIR = "results/vdbert_compare_models_layers"
os.makedirs(RESULTS_DIR, exist_ok=True)

COCO_API_URL_LOCAL = (
    "VisualDialog_val2018_"
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def download_coco_image(image_id):
    """
    Load COCO image from local dataset

    Args:
        image_id: COCO image ID

    Returns:
        PIL Image in RGB format
    """
    img_name = f"{COCO_API_URL_LOCAL}{image_id:012d}.jpg"
    return Image.open(img_name).convert("RGB")


# ============================================================================
# ATTENTION EXTRACTION
# ============================================================================


def get_all_cross_modal_attentions(
    model,
    tokenizer,
    text: str,
    visual_features: torch.Tensor,
    visual_boxes: torch.Tensor,
    device: str = "cuda",
):
    """
    Extract all cross-modal attention patterns from the model for a given question and visual features

    Args:
        model: Loaded BERT model
        tokenizer: Tokenizer (TransformersTokenizerWrapper)
        text: Question text in Portuguese or other language
        visual_features: Tensor of shape (36, 2048)
        visual_boxes: Tensor of shape (36, 4)
        device: "cuda" or "cpu"

    Returns:
        att_all: List of attention tensors (one list per layer)
        tokens: List of text tokens + visual tokens
    """
    model.eval()
    model.to(device)
    visual_features = visual_features.to(device).unsqueeze(0)  # [1, 36, 2048]

    # Text tokenization
    tokens = tokenizer.tokenize(text)
    text_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Special tokens
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # Number of visual regions
    num_vis_tokens = visual_features.shape[1]

    # Create input_ids simulating visual regions as a single token (vis_id)
    vis_id = tokenizer.unk_token_id  # use unk_token as placeholder
    input_ids = [cls_id] + [vis_id] * num_vis_tokens + [sep_id] + text_ids + [sep_id]

    # Attention mask
    attention_mask = [1] * len(input_ids)

    # Convert to tensor
    input_ids = torch.tensor([input_ids], device=device)
    attention_mask = torch.tensor([attention_mask], device=device)

    # Forward pass with attention extraction
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
        )

    att_all = outputs.attentions  # list [layer][batch, head, seq_len, seq_len]

    return att_all, tokens


def extract_cross_modal_attention(att_tensor):
    """
    Extract cross-modal attention (text tokens -> visual regions)

    Args:
        att_tensor: Attention tensor [batch, heads, seq_len, seq_len] or [heads, seq_len, seq_len]

    Returns:
        cross_att: Numpy array [36] with average attention for each visual region
    """
    # Convert to numpy
    if isinstance(att_tensor, torch.Tensor):
        att_tensor = att_tensor.cpu().detach().numpy()

    # Remove batch dimension if it exists
    if att_tensor.ndim == 4:
        att_tensor = att_tensor[0]  # [heads, seq_len, seq_len]

    # Structure: [CLS] + 36 visual regions + [SEP] + text + [SEP]
    # Visual region positions: 1 to 37 (indices 1:37)
    # Text tokens start after second SEP (position 38+)

    num_visual = 36
    visual_start = 1
    visual_end = visual_start + num_visual
    text_start = visual_end + 1  # Skip SEP after visual regions

    # Extract attention from text tokens to visual regions
    # att_tensor: [heads, seq_len, seq_len]
    # We want: attention from text_start: to visual_start:visual_end
    cross_modal_att = att_tensor[:, text_start:, visual_start:visual_end]

    # Aggregate over heads and text tokens
    cross_modal_att = cross_modal_att.mean(axis=(0, 1))  # [36]

    return cross_modal_att


def compare_attention(att1, att2):
    """
    Compare cross-modal attention between two models

    Args:
        att1: Attention tensor from model 1
        att2: Attention tensor from model 2

    Returns:
        avg1: Cross-modal attention from model 1 [36]
        avg2: Cross-modal attention from model 2 [36]
        diff: Absolute difference [36]
    """
    avg1 = extract_cross_modal_attention(att1)
    avg2 = extract_cross_modal_attention(att2)
    diff = np.abs(avg1 - avg2)
    return avg1, avg2, diff


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_comparison(image, boxes, att1, att2, diff, save_path):
    """
    Plot attention comparison between two models

    Args:
        image: PIL Image
        boxes: Numpy array [36, 4] with normalized bounding boxes
        att1: Attention tensor from model 1
        att2: Attention tensor from model 2
        diff: Difference between attentions (not used, will be recalculated)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    img_w, img_h = image.size
    boxes_abs = boxes.copy()
    boxes_abs[:, [0, 2]] *= img_w
    boxes_abs[:, [1, 3]] *= img_h

    def plot_attention(ax, title, att_scores, cmap):
        """
        Plot attention on visual regions

        Args:
            ax: Matplotlib axis
            title: Plot title
            att_scores: Array [36] with attention score per region
            cmap: Colormap for visualization
        """
        ax.imshow(image)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.axis("off")

        # Normalize scores to [0, 1]
        att_norm = (att_scores - att_scores.min()) / (
            att_scores.max() - att_scores.min() + 1e-8
        )

        # Plot boxes with color proportional to attention
        for idx in range(len(att_norm)):
            score = att_norm[idx]

            if score > 0.8:  # Threshold for visualization
                x1, y1, x2, y2 = boxes_abs[idx][:4]
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=3,
                    edgecolor=cmap(score),
                    facecolor="none",
                    alpha=0.9,
                )
                ax.add_patch(rect)

    # Extract cross-modal attention
    att1_cross = extract_cross_modal_attention(att1)
    att2_cross = extract_cross_modal_attention(att2)
    diff_cross = np.abs(att1_cross - att2_cross)

    # Plot three visualizations
    plot_attention(axes[0], "PT Model", att1_cross, plt.cm.Blues)
    plot_attention(axes[1], "Multilingual Model", att2_cross, plt.cm.Reds)
    plot_attention(axes[2], "|Difference|", diff_cross, plt.cm.Purples)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN COMPARISON
# ============================================================================


def main():
    """Main execution function"""
    print("=" * 70)
    print("VD-BERT MODEL COMPARISON (ALL LAYERS)")
    print("=" * 70)

    print("[1/6] Loading models...")
    loader, model_pt, tok_pt, _ = load_vdbert_complete(
        MODEL_PATH_PT, VISUAL_FEATURES, DATASET_PATH_PT, "pt"
    )
    loader2, model_mult, tok_mult, _ = load_vdbert_complete(
        MODEL_PATH_MULT, VISUAL_FEATURES, DATASET_PATH_ES, "es"
    )

    print("[2/6] Loading example...")
    example = loader.get_dialog_example(dialog_idx=75, turn_idx=0)
    image_id = example["image_id"]
    question = example["question"]
    caption = example.get("caption", "<no caption>")
    answer = example.get("answer", "<no answer>")
    boxes = example["visual_boxes"].numpy()
    image = download_coco_image(image_id)

    print("\nPT Example:")
    print(f"  Question: {question}")
    print(f"  Caption: {caption}")
    print(f"  Answer: {answer}")

    example2 = loader2.get_dialog_example(dialog_idx=75, turn_idx=0)
    image_id2 = example2["image_id"]
    question2 = example2["question"]
    caption2 = example2.get("caption", "<no caption>")
    answer2 = example2.get("answer", "<no answer>")
    boxes2 = example2["visual_boxes"].numpy()

    print("\nES Example:")
    print(f"  Question: {question2}")
    print(f"  Caption: {caption2}")
    print(f"  Answer: {answer2}")

    print("\n[3/6] Extracting attention from both models (all layers)...")
    att_all_pt, tokens_pt = get_all_cross_modal_attentions(
        model_pt,
        tok_pt,
        question,
        example["visual_features"],
        visual_boxes=example["visual_boxes"],
        device=loader.device,
    )
    att_all_mult, tokens_mult = get_all_cross_modal_attentions(
        model_mult,
        tok_mult,
        question2,
        example2["visual_features"],
        visual_boxes=example2["visual_boxes"],
        device=loader2.device,
    )

    print(f"\nTotal layers: {len(att_all_pt)}")
    print(f"\nPT Tokens: {tokens_pt}")
    print(f"Multilingual Tokens: {tokens_mult}")

    # Layer-wise statistics
    print("\n[4/6] Computing layer-wise statistics:")
    print(
        f"{'Layer':<8} {'Mean PT':<12} {'Mean Mult':<12} {'Mean Diff':<12} {'Max Diff':<12}"
    )
    print("-" * 60)

    for i, (a_pt, a_mult) in enumerate(zip(att_all_pt, att_all_mult)):
        avg_pt, avg_mult, diff = compare_attention(a_pt, a_mult)
        print(
            f"{i:<8} {avg_pt.mean():<12.4f} {avg_mult.mean():<12.4f} "
            f"{diff.mean():<12.6f} {diff.max():<12.6f}"
        )

    # Layer-by-layer visualization
    print("\n[5/6] Generating layer-by-layer visualizations...")
    for layer_to_plot in tqdm(range(len(att_all_pt)), desc="Creating figures"):
        save_path = os.path.join(
            RESULTS_DIR, f"compare_{image_id}_layer_{layer_to_plot:02d}.png"
        )
        plot_comparison(
            image,
            boxes,
            att_all_pt[layer_to_plot],
            att_all_mult[layer_to_plot],
            None,  # diff not used, will be recalculated internally
            save_path,
        )

    print("\n[6/6] Comparison complete!")
    print(f"Images saved in: {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
