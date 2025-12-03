# Multilingual Visual Dialog: Portuguese and Spanish

> **Training VD-BERT models for Portuguese and Spanish Visual Dialog using monolingual and multilingual encoders**

This repository provides instructions for training Visual Dialog models for Portuguese and Spanish languages, comparing monolingual (BERTimbau, BETO) and multilingual (mBERT) approaches.


## 🔍 Overview

This project extends Visual Dialog to Portuguese and Spanish by adapting VD-BERT with different language encoders:

- **Portuguese Monolingual**: [BERTimbau](https://github.com/neuralmind-ai/portuguese-bert)
- **Spanish Monolingual**: [BETO](https://github.com/dccuchile/beto)
- **Multilingual**: [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md)

**Key Results:**
- Monolingual models achieve **17-20% improvement** over multilingual
- First Visual Dialog models for Portuguese and Spanish
- Novel mechanistic interpretability analysis of cross-modal attention

---

## 📊 Dataset Setup

### Step 1: Download Portuguese and Spanish Datasets
```bash
# Create datasets directory
mkdir -p data

cd data

# Clone VisDial-PT (Portuguese)
git clone https://github.com/IMScience-PPGINF-PucMinas/visdial-pt-brazilian.git data/visdial-pt

# Clone VisDial-ES (Spanish)
git clone https://github.com/IMScience-PPGINF-PucMinas/visdial-es-spanish.git data/visdial-es
```

### Step 2: Download Visual Features

Download pre-extracted Faster R-CNN features from the original VisDial:
```bash
# Download visual features (36 regions per image)
cd data
wget https://www.dropbox.com/s/twmtutniktom7tu/features_faster_rcnn_x101_train.h5
wget https://www.dropbox.com/s/5xethqpg7dfw7pk/features_faster_rcnn_x101_val.h5

### Dataset Structure
```
data/
├── visdial-pt/
│   ├── visdial_1.0_train.json
│   ├── visdial_1.0_val.json
│   
├── visdial-es/
│   ├── visdial_1.0_train.json
│   ├── visdial_1.0_val.json
│   
└── features/
    ├── features_faster_rcnn_x101_train.h5
    └── features_faster_rcnn_x101_val.h5
```

---

## 🤖 Model Setup

### Step 1: Clone Original VD-BERT
```bash
# Clone the original VD-BERT repository
git clone https://github.com/salesforce/VD-BERT.git
cd VD-BERT
```

### Step 2: Modify for Multilingual Support

Replace the encoder initialization in ` all VD-BERT original project`:

#### For Portuguese Monolingual (BERTimbau)
```python
from transformers import AutoModel, AutoTokenizer

# Replace BERT initialization with BERTimbau
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
encoder = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
```

#### For Spanish Monolingual (BETO)
```python
from transformers import AutoModel, AutoTokenizer

# Replace BERT initialization with BETO
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
encoder = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
```


#### For Multilingual (mBERT)
```python
from transformers import BertModel, BertTokenizer

# Use multilingual BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
encoder = BertModel.from_pretrained("bert-base-multilingual-cased")
```

## 🚀 Training

### Phase 1: Initial Pretraining (30 epochs)

This is the training phase covered in our paper.

#### Portuguese with BERTimbau
```bash
python train.py \
    --config configs/train_pt_bertimbau.json \
    --train_json data/visdial-pt/visdial_1.0_train.json \
    --val_json data/visdial-pt/visdial_1.0_val.json \
    --visual_features data/features/features_faster_rcnn_x101_train.h5 \
    --encoder_type bertimbau \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --num_epochs 30 \
    --use_single_history \
    --output_dir checkpoints/pt_bertimbau_phase1
```

#### Spanish with BETO
```bash
python train.py \
    --config configs/train_es_beto.json \
    --train_json data/visdial-es/visdial_1.0_train.json \
    --val_json data/visdial-es/visdial_1.0_val.json \
    --visual_features data/features/features_faster_rcnn_x101_train.h5 \
    --encoder_type beto \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --num_epochs 30 \
    --use_single_history \
    --output_dir checkpoints/es_beto_phase1
```

#### Portuguese/Spanish with mBERT
```bash
# Portuguese
python train.py \
    --config configs/train_pt_mbert.json \
    --train_json data/visdial-pt/visdial_1.0_train.json \
    --val_json data/visdial-pt/visdial_1.0_val.json \
    --visual_features data/features/features_faster_rcnn_x101_train.h5 \
    --encoder_type mbert \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --num_epochs 30 \
    --use_single_history \
    --output_dir checkpoints/pt_mbert_phase1

# Spanish
python train.py \
    --config configs/train_es_mbert.json \
    --train_json data/visdial-es/visdial_1.0_train.json \
    --val_json data/visdial-es/visdial_1.0_val.json \
    --visual_features data/features/features_faster_rcnn_x101_train.h5 \
    --encoder_type mbert \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --num_epochs 30 \
    --use_single_history \
    --output_dir checkpoints/es_mbert_phase1
```

### Key Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 3e-4 | Optimal for Phase 1 |
| `batch_size` | 32 | Best performance vs. 64 |
| `num_epochs` | 30 | Phase 1 duration |
| `warmup_ratio` | 0.1 | Linear warmup |
| `weight_decay` | 0.01 | L2 regularization |
| `max_seq_length` | 256 | Maximum token sequence |

## 📈 Evaluation

### Discriminative Setting (Ranking)
```bash
python evaluate.py \
    --checkpoint checkpoints/pt_bertimbau_phase1/best.pth \
    --val_json data/visdial-pt/visdial_1.0_val.json \
    --visual_features data/features/features_faster_rcnn_x101_val.h5 \
    --encoder_type bertimbau \
    --output_metrics results/pt_bertimbau_metrics.json
```

### Metrics Computed

- **NDCG** (Normalized Discounted Cumulative Gain)
- **MRR** (Mean Reciprocal Rank)
- **Recall@k** (k = 1, 5, 10)
- **Mean Rank**

### Expected Results (Phase 1)

| Model | Language | NDCG↑ | MRR↑ | R@1↑ | R@5↑ | R@10↑ |
|-------|----------|-------|------|------|------|-------|
| BERTimbau | PT | 0.21 | 0.053 | 1.05 | 5.07 | 11.10 |
| mBERT | PT | 0.20 | 0.045 | 0.76 | 4.00 | 8.41 |
| BETO | ES | 0.20 | 0.049 | 0.86 | 4.58 | 9.16 |
| mBERT | ES | 0.20 | 0.041 | 0.66 | 3.33 | 6.54 |

---


## 🔬 Attention Analysis

Extract and visualize cross-modal attention patterns:
```bash
# Extract attention weights
python analyze_attention.py \
    --checkpoint checkpoints/pt_bertimbau_phase1/best.pth \
    --val_json data/visdial-pt/visdial_1.0_val.json \
    --visual_features data/features/features_faster_rcnn_x101_val.h5 \
    --output_dir attention_analysis/pt_bertimbau \
    --num_samples 100

# Visualize attention heatmaps
python visualize_attention.py \
    --attention_dir attention_analysis/pt_bertimbau \
    --images_dir data/coco_images \
    --output_dir visualizations/pt_bertimbau
```

### Attention Analysis Scripts

- `analyze_attention.py`: Extract attention weights from all layers/heads
- `visualize_attention.py`: Create attention heatmap overlays
- `compare_attention.py`: Statistical comparison between models
- `head_analysis.py`: Head-level cross-lingual analysis

---

## 📚 Project Structure
```
multilingual-vd-bert/
├── data/
│   ├── visdial-pt/           # Portuguese dataset
│   ├── visdial-es/           # Spanish dataset
│   └── features/             # Visual features
├── configs/
│   ├── train_pt_bertimbau.json
│   ├── train_es_beto.json
│   └── train_mbert.json
├── vdbert/
│   ├── modeling.py           # Modified VD-BERT model
│   ├── encoders.py           # Language encoder wrappers
│   └── data_loader.py        # Data loading utilities
|   ...
├── scripts/
│   ├── download_features.py
│   ├── download_models.py
│   └── preprocess_data.py
├── train.py                  # Main training script
├── evaluate.py               # Evaluation script
├── analyze_attention.py      # Attention analysis
├── visualize_attention.py    # Attention visualization
└── requirements.txt
```

---

## 📖 Citation

If you use this code or models in your research, please cite:
```bibtex
@inproceedings{adao2025multilingual,
  title={Extending Visual Dialog beyond English: An Analysis of Monolingual and Multilingual Models},
  author={Ad{\~a}o, Milena Menezes and Guimar{\~a}es, Silvio Jamil F. and Patroc{\'i}nio Jr, Zenilton K. G.},
  booktitle={28th Iberoamerican Congress on Pattern Recognition (CIARP)},
  year={2025}
}
```



## 🔗 Useful Links

### Datasets
- [VisDial-PT (Portuguese)](https://github.com/IMScience-PPGINF-PucMinas/visdial-pt-brazilian)
- [VisDial-ES (Spanish)](https://github.com/IMScience-PPGINF-PucMinas/visdial-es-spanish)
- [Original VisDial v1.0](https://visualdialog.org/)

### Models
- [VD-BERT Original](https://github.com/salesforce/VD-BERT)
- [BERTimbau (Portuguese)](https://github.com/neuralmind-ai/portuguese-bert)
  - HF: [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
- [BETO (Spanish)](https://github.com/dccuchile/beto)
  - HF: [dccuchile/bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)
- [mBERT (Multilingual)](https://github.com/google-research/bert/blob/master/multilingual.md)
  - HF: [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)

### Visual Features
- [Faster R-CNN Features](https://github.com/peteanderson80/bottom-up-attention)

---


## 📧 Contact

**Milena Menezes Adão**
- Email: milena.adao@sga.pucminas.br

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.





Made with ❤️ by [IMScience Lab](https://github.com/IMScience-PPGINF-PucMinas)

</div>
