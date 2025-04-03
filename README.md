# RecognizingHandwrittenDigits

This project uses AI to recognize and classify handwritten digits using the MNIST dataset. It allows users to train and test a neural network with customizable parameters and upload their own digit images for classification. The code is based off of existing code and updated to work with python3.

---

# Project Structure

```
RecognizingHandwrittenDigits/
├── data/
│   └── mnist.pkl.gz               # MNIST dataset file (must remain in this location)
├── fig/                           # Directory for generated figures and plots
├── src/                           # Source code (helper functions, modules)
├── HandWrittenDigitsAIProject.py # Main script to train and test the model
├── requirements.txt              # Dependency versions
└── README.md
```

> **⚠️ Important:**  
> The provided file structure is essential for running the code properly. Do **not** change folder or file locations (especially `data/mnist.pkl.gz`) or the program may not work correctly.

---

# Requirements

The following Python packages are required (installed via `requirements.txt`):

```
numpy==1.13.3  
scikit-learn==0.19.0  
scipy==0.19.1  
Theano==0.7.0
```

# Install Dependencies

Use the following command to install all required packages:

```bash
pip3 install -r requirements.txt
```

---

# Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/NathanielYC/RecognizingHandwrittenDigits.git
cd RecognizingHandwrittenDigits
```

2. **Train and test the model**

```bash
python3 HandWrittenDigitsAIProject.py
```

---

# Dataset

This project uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), loaded from a pre-downloaded `mnist.pkl.gz` file located in the `data/` folder.

---

# Image Upload Requirements

You can test your own handwritten digit images. Make sure the images meet the following:

- **Size:** 28x28 pixels  
- **Format:** Compatible with your preprocessing (grayscale recommended)

---

# Features

- Train a neural network on the MNIST dataset  
- Fixed structure tied to dataset and script organization  
- Upload custom digit images for classification  
- Easy-to-run single script interface

---

# License

This project is open source under the [MIT License](LICENSE).

---

# Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)  
- Scikit-learn, NumPy, SciPy, and Theano documentation
