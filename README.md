# 🧠 Brain Tumor MRI Classification using Knowledge Distillation

A PyTorch implementation of **knowledge distillation** for brain tumor classification, where a lightweight **MobileNetV3-Small** student model learns from a powerful **ResNet-101** teacher model.

---

## 📋 Overview

This project demonstrates how knowledge distillation can compress a large neural network into a smaller, more efficient model while maintaining high accuracy.
The system classifies brain MRI images into four categories:

* Glioma
* Meningioma
* No Tumor
* Pituitary Tumor

### ✨ Key Features

* 🧠 **Teacher Model**: ResNet-101 (pre-trained on ImageNet)
* 📱 **Student Model**: MobileNetV3-Small (pre-trained on ImageNet)
* 🔥 **Knowledge Distillation**: Temperature-scaled soft targets + hard loss
* 🎨 **Visualization**: Grad-CAM heatmaps to compare teacher vs student attention
* 📈 **High Performance**: Student achieves **98.7%** accuracy on the test set

---

## 🎯 Results

| Model                       | Accuracy | F1 Score |
| --------------------------- | -------- | -------- |
| Teacher (ResNet-101)        | 99.30%   | -        |
| Student (MobileNetV3-Small) | 98.70%   | 0.986    |

### 📉 Confusion Matrix

|                | Glioma | Meningioma | NoTumor | Pituitary |
| -------------- | ------ | ---------- | ------- | --------- |
| **Glioma**     | 288    | 11         | 0       | 1         |
| **Meningioma** | 1      | 303        | 2       | 0         |
| **NoTumor**    | 0      | 0          | 405     | 0         |
| **Pituitary**  | 0      | 2          | 0       | 298       |

---

## 🚀 Getting Started

### 📦 Prerequisites

```bash
pip install torch torchvision
pip install scikit-learn matplotlib seaborn tqdm grad-cam
```

---

### 📁 Dataset Structure

```plaintext
brain-tumor-mri-dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## 🏗️ Training

### Train the Teacher Model:

```bash
python train_teacher(epochs=10)
```

### Train the Student Model with Knowledge Distillation:

```bash
python train_student(epochs=10)
```

### Evaluate the Student Model:

```bash
python evaluate(student, test_loader)
```

---

## 🔬 Methodology

### 📘 Knowledge Distillation Loss

The distillation loss combines **hard** and **soft** components:

```python
loss = α * hard_loss + (1 - α) * soft_loss
```

* **Hard Loss**: Cross-entropy with true labels
* **Soft Loss**: KL divergence between student and teacher predictions
* **Temperature (T=4)**: Softens probability distributions
* **Alpha (α=0.5)**: Balances hard and soft losses

---

### 🧪 Data Augmentation

* Random horizontal flip
* Random rotation (±15°)
* Resize to 224×224
* ImageNet normalization

---

## 📊 Visualizations

The project includes visualization tools to better understand model performance:

* 📈 Prediction Comparison: Bar charts for teacher vs student confidence
* 📉 Confusion Matrix: Heatmap of classification performance
* 🔥 Grad-CAM: Visualizes attention regions in MRI scans

Example usage:

```python
visualize_classification_kd(teacher, student, val_dataset, idx=1000, device='cuda')
```

---

## 🏗️ Architecture Details

### 🧠 Teacher Network (ResNet-101)

* Depth: 101 layers
* Parameters: ~44M
* Final layer modified for 4-class classification

### 📱 Student Network (MobileNetV3-Small)

* Lightweight architecture
* Parameters: ~2.5M (**94% reduction**)
* Final layer modified for 4-class classification

---

## 📈 Training Configuration

```python
Batch Size: 32
Learning Rate: 1e-4
Optimizer: Adam
Teacher Epochs: 10
Student Epochs: 10
Validation Split: 20%
```

---

## 🔍 Key Functions

* `train_teacher()` – Train the ResNet-101 teacher model
* `train_student()` – Train the student using KD
* `kd_loss()` – Compute combined distillation loss
* `evaluate()` – Calculate model accuracy
* `visualize_classification_kd()` – Visualize predictions with Grad-CAM

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}
```

---

## 📄 License

This project is available for **educational and research purposes** under the MIT License.

---

## 🙏 Acknowledgments

* 🧬 Brain Tumor MRI Dataset from [Kaggle](https://www.kaggle.com/)
* 🧠 PyTorch and torchvision teams
* 🔥 Grad-CAM implementation from [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](../../issues).
