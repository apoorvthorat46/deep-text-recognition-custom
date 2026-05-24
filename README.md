# Medical Sonography Text Recognition — Fine-Tuned STR Model
| [Original Paper](https://arxiv.org/abs/1904.01906) | [Original Repository](https://github.com/clovaai/deep-text-recognition-benchmark) | [Pretrained Model](./saved_models/Final_Sonography_Model/custom_model.pth) |

This repository is a **fine-tuned fork** of the [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) framework, adapted for **medical ultrasound/sonography text recognition**.

The original four-stage STR framework (Transformation → Feature Extraction → Sequence Modeling → Prediction) is retained, with the model fine-tuned on a custom dataset of **sonography examination images** to extract text such as patient information, anatomical measurements, medical findings, and technical parameters from ultrasound scan reports.

The system integrates with [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) for text detection and provides a complete end-to-end pipeline that generates structured **JSON medical reports**.

## Model Architecture

**TRBA (TPS-ResNet-BiLSTM-Attn)** — the best accuracy configuration from the original framework:

| Stage | Module |
|-------|--------|
| Transformation | TPS (Thin-Plate Spline) |
| Feature Extraction | ResNet |
| Sequence Modeling | BiLSTM |
| Prediction | Attention |

**Model details:**
- Input size: 32 × 100 (grayscale)
- Character set: `0123456789abcdefghijklmnopqrstuvwxyz` (36 characters)
- Batch max length: 110
- Hidden size: 256
- Output channels: 512
- Num fiducial points: 20

## Updates

**May 2026**: Fine-tuned model for sonography text recognition with full end-to-end pipeline (CRAFT detection + OCR + medical report generation). <br>
Based on the original framework from [clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

## Getting Started

### Dependencies
- Python 3.6+
- PyTorch 1.3.1+ / CUDA 10.1+
- Additional requirements:
```
pip3 install lmdb pillow torchvision nltk natsort opencv-python fire
```

### Project Structure

```
deep-text-recognition-benchmark/
├── custom_model.py              # Model definition (TRBA architecture)
├── simple_ocr.py                # Simple single-image OCR inference
├── full_pipeline.py             # End-to-end pipeline (detection + OCR + report)
├── utils.py                     # Label converters (CTC, Attention)
├── dataset.py                   # Dataset loader
├── modules/
│   ├── transformation.py        # TPS spatial transformer
│   ├── feature_extraction.py    # VGG, RCNN, ResNet extractors
│   ├── sequence_modeling.py     # BiLSTM
│   └── prediction.py            # CTC, Attention
├── CRAFT-pytorch/               # Text detection module
└── saved_models/
    └── Final_Sonography_Model/  # Fine-tuned model weights
        ├── custom_model.pth     # Trained model weights
        ├── custom_model.yaml    # Model configuration
        └── opt.txt              # Training options
```

### Run Simple OCR (Single Cropped Text Image)

For recognizing text from a single cropped word image:

```
python3 simple_ocr.py --image path/to/cropped_text.png --model saved_models/Final_Sonography_Model/custom_model.pth
```

### Run Full End-to-End Pipeline (Detection + OCR + Report)

The full pipeline uses CRAFT for text detection and the fine-tuned OCR model for recognition, producing a structured JSON medical report.

**Process a single image:**
```
python3 -c "
from full_pipeline import process_single_image
results, error = process_single_image('path/to/ultrasound.jpg', 'CRAFT-pytorch/craft_mlt_25k.pth', 'saved_models/Final_Sonography_Model/custom_model.pth')
if results:
    text_report, json_data = generate_medical_report(results, return_json=True)
    print(text_report)
"
```

**Process a batch of images:**
```
python3 -c "
from full_pipeline import process_batch
process_batch('path/to/images/', 'CRAFT-pytorch/craft_mlt_25k.pth', 'saved_models/Final_Sonography_Model/custom_model.pth', './reports/')
"
```

### Training on Custom Medical Dataset

1. Prepare your dataset in LMDB format:
```
pip3 install fire
python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/
```

The ground truth file (`gt.txt`) should follow this format:
```
{imagepath}\t{label}\n
```
Example:
```
test/word_1.png uterus
test/word_2.png 140 cm
test/word_3.png CA17S
...
```

2. Fine-tune the model:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--train_data ./lmdb_final_3600 --valid_data ./lmdb_final_3600 \
--select_data lmdb_final_3600 --batch_ratio 1.0 \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--batch_max_length 110 \
--data_filtering_off \
--saved_model TPS-ResNet-BiLSTM-Attn.pth
```

3. Evaluate the model:
```
CUDA_VISIBLE_DEVICES=0 python3 test.py \
--eval_data ./lmdb_final_3600 --benchmark_all_eval \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model saved_models/Final_Sonography_Model/custom_model.pth
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--train_data` | Folder path to training LMDB dataset |
| `--valid_data` | Folder path to validation LMDB dataset |
| `--eval_data` | Folder path to evaluation LMDB dataset |
| `--select_data` | Select training data source(s) |
| `--batch_ratio` | Ratio for each selected data in the batch |
| `--data_filtering_off` | Skip data filtering when creating LmdbDataset |
| `--Transformation` | Transformation module [None \| TPS] |
| `--FeatureExtraction` | Feature extraction module [VGG \| RCNN \| ResNet] |
| `--SequenceModeling` | Sequence modeling module [None \| BiLSTM] |
| `--Prediction` | Prediction module [CTC \| Attn] |
| `--saved_model` | Path to saved model for evaluation/continued training |
| `--benchmark_all_eval` | Evaluate on all benchmark datasets |
| `--batch_max_length` | Maximum text length per batch |

## Dataset

The model was fine-tuned on a custom LMDB dataset (`lmdb_final_3600`) consisting of cropped text regions from **sonography/ultrasound examination images**. The dataset includes text from:

- Patient information (names, ages, dates)
- Anatomical measurements (cm, mm, ml, hz, mhz)
- Medical findings and anatomical terms
- Technical parameters (ultrasound machine settings)
- Center/diagnostic center information

## Report Generation

The end-to-end pipeline (`full_pipeline.py`) produces structured JSON reports with the following categories:

```json
{
  "patient_info": {
    "name": "Ashwini Patils",
    "age": "25 years",
    "date": "27/02/2024",
    "doctor": "Dr. Deepali"
  },
  "measurements": {
    "length_dimensions": ["140 cm", "13 cm"],
    "volume": ["557 ml"],
    "frequency": ["55 hz", "21 mhz"]
  },
  "findings": ["uterus", "ovary", "CA17S", "MIF"],
  "technical": {
    "ultrasound_machine": "Samsung",
    "scan_mode": "2D"
  },
  "center_info": ["Disha Diagnostics Centre"]
}
```

## Performance

- **Model architecture**: TRBA (TPS-ResNet-BiLSTM-Attn)
- **Overall accuracy**: ~90-95% on sonography text
- **Processing speed**: ~7s per full ultrasound image (detection + OCR)
- **Batch capability**: Multiple images processed sequentially
- **Report format**: Structured JSON medical documentation

## Acknowledgements

This work is based on the following repositories and frameworks:

- [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) — Original STR framework by Clova AI (NAVER Corp.)
- [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) — Character Region Awareness for Text Detection
- [crnn.pytorch](https://github.com/meijieru/crnn.pytorch)
- [ocr_attention](https://github.com/marvis/ocr_attention)

## Reference

[1] J. Baek, G. Kim, J. Lee, S. Park, D. Han, S. Yun, S. J. Oh, and H. Lee. What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis. In ICCV, 2019.

[2] B. Shi, X. Bai, and C. Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. In TPAMI, volume 39, pages 2298–2304, 2017.

[3] Y. Baek, B. Lee, D. Han, S. Yun, and H. Lee. Character Region Awareness for Text Detection. In CVPR, 2019.

## Citation

If you find this work useful for your research, please cite the original paper:

```
@inproceedings{baek2019STRcomparisons,
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019},
  pubstate={published},
  tppubtype={inproceedings}
}
```

## License

Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.