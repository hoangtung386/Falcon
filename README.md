# Falcon

Multi-input Transformer for Age and Gender Estimation.

Falcon is a computer vision model that estimates a person's age and gender from images. It uses a multi-input transformer architecture that can jointly process face crops and whole-body crops for improved accuracy.

## Features

- **Age estimation** (regression / distribution / ordinal) and **gender classification**
- **Face-only** or **face+body** modes with **gated cross-attention fusion**
- Overlapping patch embedding for better boundary information retention
- YOLOv8-based face and person detection
- Half-precision (FP16) inference
- Dataset preparation tools for UTK, FairFace, Adience, AgeDB, CACD, LAGENDA
- Full training pipeline with AdamW, AMP, CosineAnnealingLR

## Project Structure

```
Falcon/
├── falcon/                            # Main package
│   ├── __init__.py                    # Package entry point
│   ├── config.py                      # ModelConfig dataclass
│   ├── losses.py                      # AgeGenderLoss, OrdinalAgeLoss, WeightedMSE
│   ├── predictor.py                   # High-level Predictor (detection + age/gender)
│   ├── eval_metrics.py                # Metrics, time_sync, write_results
│   ├── version.py                     # __version__
│   ├── structures/                    # Detection result containers
│   │   ├── __init__.py
│   │   ├── types.py                   # AGE_GENDER_TYPE alias
│   │   ├── crops.py                   # PersonAndFaceCrops
│   │   └── result.py                  # PersonAndFaceResult
│   ├── model/                         # Model definitions
│   │   ├── __init__.py
│   │   ├── falcon_model.py            # FalconModel (VOLO-based with dual-branch)
│   │   ├── cross_attention.py         # CrossBottleneckAttn with Gated Fusion
│   │   ├── factory.py                 # Checkpoint loading, model creation
│   │   ├── inference.py               # Falcon inference wrapper
│   │   └── yolo_detector.py           # YOLOv8 face+person detector
│   └── data/                          # Data pipelines
│       ├── __init__.py
│       ├── io.py                      # PictureInfo, InputType, CSV parsing
│       ├── transforms/                # Image preprocessing
│       │   ├── __init__.py
│       │   ├── image.py               # class_letterbox, prepare_images
│       │   ├── geometry.py            # box_iou, assign_faces
│       │   └── metrics.py             # aggregate_votes, cumulative_score/error
│       └── dataset/                   # PyTorch datasets
│           ├── __init__.py
│           ├── dataset.py             # AgeGenderDataset (regression)
│           ├── classification.py      # FairFaceDataset, AdienceDataset
│           ├── loader.py              # PrefetchLoader, create_loader
│           └── reader.py              # ReaderAgeGender, CSV parsing
├── tools/                             # Dataset preparation scripts
│   ├── __init__.py
│   ├── dataset_visualization.py
│   ├── download_lagenda.py
│   ├── preparation_utils.py
│   ├── prepare_adience.py
│   ├── prepare_agedb.py
│   ├── prepare_cacd.py
│   └── prepare_fairface.py
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_data_reader.py
│   ├── test_losses.py
│   ├── test_misc.py
│   ├── test_structures.py
│   └── test_transforms.py
├── train.py                           # Training pipeline
├── eval_pretrained.py                 # Evaluation on benchmark datasets
├── setup.py                           # pip-installable package
├── requirements.txt                   # Python dependencies
├── setup.cfg                          # pytest and metadata config
├── mypy.ini                           # Type checking config
├── .flake8                            # PEP8 linting config
├── .isort.cfg                         # Import sorting config
└── .gitignore                         # Git exclusions
```

## Installation

```bash
pip install -e .
```

Requires Python 3.8+, PyTorch 1.13+, and CUDA-capable GPU (optional but recommended).

Dependencies are in `requirements.txt`.

## Usage

### Evaluation

```bash
python eval_pretrained.py \
    --dataset_images data/utk/images \
    --dataset_annotations data/utk/annotation \
    --dataset_name utk \
    --checkpoint pretrained/checkpoint-377.pth.tar \
    --batch-size 512 --half --with-persons
```

### Training

```bash
python train.py \
    --dataset-images data/lagenda/images \
    --dataset-annotations data/lagenda/annotation \
    --dataset-name lagenda \
    --checkpoint pretrained/falcon_d1_224.pth.tar \
    --batch-size 64 --epochs 100 --lr 1e-4 \
    --half --with-persons \
    --age-mode distribution
```

### Tests

```bash
python -m pytest tests/ -v
```

## Development

### Code quality

The project uses the following tools:

| Tool | Purpose | Config |
|------|---------|--------|
| **black** | Code formatter | `--line-length 120` |
| **isort** | Import sorting | `--profile black` |
| **flake8** | PEP8 linting | `.flake8` |
| **mypy** | Static type checking | `mypy.ini` |

```bash
# Format code
black --line-length 120 falcon/ tools/ tests/ train.py eval_pretrained.py
isort --profile black --line-length 120 falcon/ tools/ tests/ train.py eval_pretrained.py
# Lint
flake8 --config .flake8 falcon/ tools/ tests/ train.py eval_pretrained.py
# Type check
mypy falcon/ train.py eval_pretrained.py
```

### Future improvements

Planned technical improvements:

- **Sprint 1**: Adaptive loss, long-tail handling, curriculum learning
- **Sprint 2**: Bidirectional cross-attention, multi-task auxiliary heads
- **Sprint 3**: DINOv2/ConvNeXt-ViT hybrid backbone, deformable patch embedding

## Datasets

| Dataset | Task | Script |
|---------|------|--------|
| UTKFace | Age regression + gender | Built-in |
| LAGENDA | Age regression + gender | `tools/download_lagenda.py` |
| FairFace | Age classification + gender | `tools/prepare_fairface.py` |
| Adience | Age classification + gender | `tools/prepare_adience.py` |
| AgeDB | Age regression + gender | `tools/prepare_agedb.py` |
| CACD | Age regression + gender | `tools/prepare_cacd.py` |
| IMDB-cleaned | Age regression + gender | Built-in |

## References

- **MLLM 2024** — Assessing MLLMs for age/gender estimation — [arXiv 2403.02302](https://arxiv.org/abs/2403.02302)
- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)

## License

[MIT](./LICENSE)
