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
DATASET_PATH2 = "dataset/visdial_1.0_val-ES.json"
VISUAL_FEATURES = "img_feats1.0/train.h5"
DATASET_PATH = "dataset/visdial_1.0_val-PT.json"

RESULTS_DIR = "results/vdbert_compare_models_layers"
os.makedirs(RESULTS_DIR, exist_ok=True)

COCO_API_URL_local = (
    "VisualDialog_val2018_"
)


# ============================================================================
# FUNCTIONS
# ============================================================================


def download_coco_image(image_id):
    """Load a COCO image from local storage."""
    img_name = f"{COCO_API_URL_local}{image_id:012d}.jpg"
    return Image.open(img_name).convert("RGB")


def get_all_cross_modal_attentions(
    model,
    tokenizer,
    text: str,
    visual_features: torch.Tensor,
    visual_boxes: torch.Tensor,
    device: str = "cuda",
):
    """
    Generate all cross-modal attention maps for a given text and visual features.

    Args:
        model: Loaded BERT model.
        tokenizer: Tokenizer (TransformersTokenizerWrapper).
        text: Input question (Portuguese or any language).
        visual_features: Tensor of shape (36, 2048).
        visual_boxes: Tensor of shape (36, 4).
        device: "cuda" or "cpu".

    Returns:
        att_all: List of attention tensors (one per layer).
        tokens: Tokenized text.
    """
    model.eval()
    model.to(device)
    visual_features = visual_features.to(device).unsqueeze(0)

    # Tokenization
    tokens = tokenizer.tokenize(text)
    text_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Special tokens
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # Visual token count
    num_vis_tokens = visual_features.shape[1]

    # Use unk_token as placeholder for visual tokens
    vis_id = tokenizer.unk_token_id
    input_ids = [cls_id] + [vis_id] * num_vis_tokens + [sep_id] + text_ids + [sep_id]

    # Attention mask
    attention_mask = [1] * len(input_ids)

    # Convert to tensor
    input_ids = torch.tensor([input_ids], device=device)
    attention_mask = torch.tensor([attention_mask], device=device)

    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
        )

    att_all = outputs.attentions
    return att_all, tokens


def extract_cross_modal_attention(att_tensor):
    """
    Extract cross-modal attention (text tokens → visual regions).

    Args:
        att_tensor: Attention tensor [batch, heads, seq_len, seq_len]
                    or [heads, seq_len, seq_len].

    Returns:
        cross_att: Numpy array [36] with average attention per visual region.
    """
    if isinstance(att_tensor, torch.Tensor):
        att_tensor = att_tensor.cpu().detach().numpy()

    # Remove batch dimension if present
    if att_tensor.ndim == 4:
        att_tensor = att_tensor[0]

    # Visual regions: positions 1 to 36
    num_visual = 36
    visual_start = 1
    visual_end = visual_start + num_visual

    # Text tokens start after the visual SEP
    text_start = visual_end + 1

    # Extract cross-modal attention
    cross_modal_att = att_tensor[:, text_start:, visual_start:visual_end]

    # Mean across heads and text tokens
    cross_modal_att = cross_modal_att.mean(axis=(0, 1))
    return cross_modal_att


def compare_attention(att1, att2):
    """
    Compare cross-modal attentions between two models.

    Returns:
        avg1: Model 1 attention [36].
        avg2: Model 2 attention [36].
        diff: Absolute difference [36].
    """
    avg1 = extract_cross_modal_attention(att1)
    avg2 = extract_cross_modal_attention(att2)
    diff = np.abs(avg1 - avg2)
    return avg1, avg2, diff


def plot_comparison(image, boxes, att1, att2, diff, save_path):
    """
    Plot comparison of two models' attention maps.

    Args:
        image: PIL image.
        boxes: Numpy array [36, 4] with normalized bounding boxes.
        att1: Attention from model 1.
        att2: Attention from model 2.
        diff: Unused (recomputed internally).
        save_path: Output file path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    img_w, img_h = image.size

    boxes_abs = boxes.copy()
    boxes_abs[:, [0, 2]] *= img_w
    boxes_abs[:, [1, 3]] *= img_h

    def plot_attention(ax, title, att_scores, cmap):
        """Plot attention for a visual region set."""
        ax.imshow(image)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.axis("off")

        att_norm = (att_scores - att_scores.min()) / (
            att_scores.max() - att_scores.min() + 1e-8
        )

        for idx in range(len(att_norm)):
            score = att_norm[idx]

            if score > 0.8:
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

    # Compute attention values
    att1_cross = extract_cross_modal_attention(att1)
    att2_cross = extract_cross_modal_attention(att2)
    diff_cross = np.abs(att1_cross - att2_cross)

    # Plot
    plot_attention(axes[0], "Model PT", att1_cross, plt.cm.Blues)
    plot_attention(axes[1], "Multilingual Model", att2_cross, plt.cm.Reds)
    plot_attention(axes[2], "|Difference|", diff_cross, plt.cm.Purples)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN LOGIC
# ============================================================================


def main():
    print("=" * 70)
    print("VD-BERT MODEL COMPARISON (ALL LAYERS)")
    print("=" * 70)

    print("[1/6] Loading models...")
    loader, model_pt, tok_pt, _ = load_vdbert_complete(
        MODEL_PATH_PT, VISUAL_FEATURES, DATASET_PATH, "pt"
    )
    loader2, model_mult, tok_mult, _ = load_vdbert_complete(
        MODEL_PATH_MULT, VISUAL_FEATURES, DATASET_PATH2, "es"
    )

    print("[2/6] Loading example...")
    example = loader.get_dialog_example(dialog_idx=75, turn_idx=0)
    image_id = example["image_id"]
    question = example["question"]
    caption = example.get("caption", "<no caption>")
    answer = example.get("answer", "<no answer>")
    boxes = example["visual_boxes"].numpy()
    image = download_coco_image(image_id)

    print("\n🗨 Question:", question)
    print("🖼 Caption:", caption)
    print("💬 Answer:", answer)

    example2 = loader2.get_dialog_example(dialog_idx=75, turn_idx=0)
    image_id2 = example2["image_id"]
    question2 = example2["question"]
    caption2 = example2.get("caption", "<no caption>")
    answer2 = example2.get("answer", "<no answer>")
    boxes2 = example2["visual_boxes"].numpy()
    image2 = download_coco_image(image_id)

    print("\n🗨 Question:", question2)
    print("🖼 Caption:", caption2)
    print("💬 Answer:", answer2)

    print("[3/6] Extracting attention from both models...")
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
    print("\n🧩 Tokens (PT):", tokens_pt)
    print("🧩 Tokens (Multilingual):", tokens_mult)

    # Layer statistics
    print("\n📊 Layer statistics:")
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

    # Visualization
    print("\n[6/6] Generating visualizations...")
    for layer_to_plot in tqdm(range(len(att_all_pt)), desc="Generating figures"):
        save_path = os.path.join(
            RESULTS_DIR, f"compare_{image_id}_layer_{layer_to_plot:02d}.png"
        )
        plot_comparison(
            image,
            boxes,
            att_all_pt[layer_to_plot],
            att_all_mult[layer_to_plot],
            None,
            save_path,
        )

    print("\n✅ Comparison finished!")
    print(f"📁 Saved images in: {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
