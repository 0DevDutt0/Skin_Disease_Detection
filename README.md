# 🩺 Skin Disease Prediction System

A deep learning-powered web application for automated skin disease classification using ResNet50V2 architecture with explainable AI (LIME) to provide visual explanations for predictions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Diseases](#supported-diseases)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a state-of-the-art deep learning model for classifying skin diseases from images. The system uses transfer learning with ResNet50V2 and achieves high accuracy through fine-tuning. Additionally, it provides explainable AI visualizations using LIME (Local Interpretable Model-agnostic Explanations) to help users understand which parts of the image influenced the prediction.

## ✨ Features

- **High Accuracy**: ResNet50V2-based model with fine-tuning for optimal performance
- **Explainable AI**: LIME integration provides visual explanations highlighting important image regions
- **Web Interface**: User-friendly Flask web application for easy image upload and analysis
- **Real-time Predictions**: Fast inference with confidence scores
- **Image Metrics**: Additional analysis including sharpness and color distribution
- **Multiple Format Support**: Accepts various image formats (PNG, JPG, JPEG, BMP, TIFF, WebP, etc.)

## 🏥 Supported Diseases

The model can classify the following skin conditions:

1. **Acne** - Inflammatory skin condition with pimples and lesions
2. **Eczema** - Chronic inflammatory skin condition causing itchy, red patches
3. **Psoriasis** - Autoimmune condition causing scaly, red skin patches
4. **Vitiligo** - Loss of skin pigmentation resulting in white patches
5. **Warts** - Small, rough growths caused by viral infection

## 📁 Project Structure

```
Skin Disease Prediction/
├── app.py                      # Flask web application
├── train.py                    # Model training script
├── explain.py                  # LIME explanation generator
├── preprocess.py               # Data preprocessing utilities
├── eda.py                      # Exploratory data analysis
├── test_inference.py           # Model inference testing
├── verify.py                   # Model verification script
├── verify_resize.py            # Image resize verification
├── cleanup.py                  # Cleanup utility
├── tests.py                    # Unit tests
├── requirements.txt            # Python dependencies
├── class_indices.json          # Class label mappings
├── skin_disease_model_v2.h5    # Trained ResNet50V2 model
├── templates/
│   └── index.html              # Web interface template
├── static/
│   └── explanations/           # Generated LIME explanations
├── temp_uploads/               # Temporary upload directory
└── SkinDisease/                # Training dataset
    ├── Train/                  # Training images
    └── Test/                   # Testing images
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/skin-disease-prediction.git
cd skin-disease-prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python verify.py
```

## 💻 Usage

### Running the Web Application

1. **Start the Flask server:**

```bash
python app.py
```

2. **Open your browser and navigate to:**

```
http://localhost:5000
```

3. **Upload an image:**
   - Click the upload button
   - Select a skin disease image
   - View the prediction results and LIME explanation

### Training the Model

To train the model from scratch:

```bash
python train.py
```

**Note:** You need to have the dataset in the `SkinDisease/` directory with the following structure:

```
SkinDisease/
├── Train/
│   ├── Acne/
│   ├── Eczema/
│   ├── Psoriasis/
│   ├── Vitiligo/
│   └── Warts/
└── Test/
    ├── Acne/
    ├── Eczema/
    ├── Psoriasis/
    ├── Vitiligo/
    └── Warts/
```

### Testing Inference

```bash
python test_inference.py
```

### Running Tests

```bash
python tests.py
```

## 🧠 Model Architecture

### ResNet50V2 with Fine-Tuning

The model uses a two-phase training approach:

**Phase 1: Transfer Learning**
- Pre-trained ResNet50V2 (ImageNet weights)
- Frozen base layers
- Custom classification head with:
  - Global Average Pooling
  - Dense layer (256 units, ReLU)
  - Dropout (0.5)
  - Output layer (5 classes, Softmax)

**Phase 2: Fine-Tuning**
- Unfreezing top 50 layers
- Low learning rate (1e-5)
- Aggressive data augmentation
- Early stopping and learning rate reduction

### Training Configuration

- **Input Size**: 224×224×3
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Data Augmentation**: Rotation, shifts, shear, zoom, horizontal flip

### Explainability

- **LIME (Local Interpretable Model-agnostic Explanations)**
  - Generates visual heatmaps showing important image regions
  - Helps understand model decision-making
  - Increases trust and transparency

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow 2.x, Keras
- **Web Framework**: Flask
- **Image Processing**: OpenCV, scikit-image
- **Explainable AI**: LIME
- **Data Manipulation**: NumPy
- **Visualization**: Matplotlib

## 📊 Model Performance

The model achieves high accuracy through:
- Transfer learning from ImageNet
- Fine-tuning on domain-specific data
- Aggressive data augmentation
- Regularization techniques (Dropout)
- Learning rate scheduling

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ using TensorFlow and Flask**
