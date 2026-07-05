# 🔌 PCB Component Detection — YOLOv8

Detect and localize 23 types of electronic components on printed circuit boards using a custom-trained YOLOv8 model.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-purple?logo=yolo)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)

<img width="1280" height="1280" alt="WhatsApp Image 2026-01-22 at 10 45 35 PM" src="https://github.com/user-attachments/assets/27b175a1-1e33-47bd-9f8a-a0b4fbe24402" />

## ✨ Features

- 🎯 **23-class detection** — resistors, capacitors, ICs, diodes, connectors, transistors, jumpers, test points, and more
- 📊 **Automated dataset analysis** — split verification and per-class instance distribution before training
- ⚙️ **Config-driven training** — single YAML links Ultralytics to train/valid/test splits
- 🖼️ **Annotation visualizer** — sanity-checks bounding boxes against labels pre-training
- 🚀 **YOLOv8n baseline** — fast to iterate on, swappable for `s` / `m` / `l` / `x` variants
- 📈 **Built-in evaluation** — mAP50, mAP50-95, precision/recall, confusion matrix
- 🔍 **One-command inference** — run the trained model on any new PCB image

## 📁 Dataset

| Split | Images | Labels |
|---|---|---|
| Train | 548 | 548 |
| Valid | 80 | 80 |
| Test | 44 | 44 |
| **Total** | **672** | **672** |

Source: [Roboflow 100 — Printed Circuit Board Dataset](https://universe.roboflow.com/roboflow-100/printed-circuit-board/dataset/4), YOLO format (normalized `class_id x_center y_center width height`).

<details>
<summary><b>23 component classes</b></summary>

| ID | Class | ID | Class | ID | Class |
|---|---|---|---|---|---|
| 0 | Button | 8 | Ferrite Bead | 16 | Resistor Network |
| 1 | Capacitor Jumper | 9 | IC | 17 | Resistor |
| 2 | Capacitor | 10 | Inductor | 18 | Switch |
| 3 | Clock | 11 | Jumper | 19 | Test Point |
| 4 | Connector | 12 | Led | 20 | Transistor |
| 5 | Diode | 13 | Pads | 21 | Unknown Unlabeled |
| 6 | EM | 14 | Pins | 22 | iC |
| 7 | Electrolytic Capacitor | 15 | Resistor Jumper | | |

</details>

## 🗂️ Project Structure

```
circuit-component-detection-yolov8/
├── Detection.ipynb          # End-to-end training & inference notebook
├── circuit_dataset/
│   ├── data.yaml
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/
│   └── test/{images,labels}/
├── assets/                  # Result graphs & sample detection images
└── README.md
```

## 🚀 Quickstart

```bash
git clone https://github.com/TalhaAlvi1/circuit-component-detection-yolov8.git
cd circuit-component-detection-yolov8
pip install ultralytics opencv-python matplotlib
```

Open `Detection.ipynb` in Google Colab or Jupyter and run top to bottom, or use the core snippet:

```python
from ultralytics import YOLO

# Train
model = YOLO('yolov8n.pt')
model.train(data='circuit_dataset/data.yaml', epochs=100, imgsz=640, batch=16, patience=50)

# Evaluate on held-out test set
best_model = YOLO('runs/detect/train/weights/best.pt')
metrics = best_model.val(data='circuit_dataset/data.yaml', split='test')

# Run inference
results = best_model.predict(source='circuit_dataset/test/images', conf=0.25, iou=0.45, save=True)
```

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Base model | `yolov8n.pt` |
| Epochs | 100 |
| Image size | 640 |
| Batch size | 16 |
| Early stopping patience | 50 |
| Device | Auto (GPU if available, else CPU) |

## 📊 Results

<img width="1481" height="609" alt="image" src="https://github.com/user-attachments/assets/a6475dba-a4cb-4f62-a192-3de2f27730fe" />

## 🔭 Notes & Future Work

- Several classes (`Unknown Unlabeled`, duplicate `IC` / `iC`) reflect noise in the source annotations — worth cleaning up for a taxonomy pass.
- Class distribution is heavily imbalanced (e.g. Resistor Network >37k instances vs. Ferrite Bead ~108) — consider class-weighted loss or targeted augmentation.
- A larger backbone (YOLOv8m/l) and higher epoch budget should improve mAP50-95 on visually dense boards.


## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---
