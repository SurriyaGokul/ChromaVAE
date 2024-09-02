# Grayscale to Color Image Conversion using Autoencoder

This repository contains the implementation of an autoencoder model that converts grayscale images to color using self-supervised learning techniques. The model is designed to automatically colorize grayscale images with a focus on improving the vibrancy and accuracy of different colors.

## Project Overview

In this project, we developed an autoencoder-based neural network that transforms grayscale images into their corresponding color versions. The model leverages a combination of standard reconstruction loss, perceptual loss to capture high-level features, ensuring the generated colors align with perceptual reality and a custom color loss to enhance the quality of the generated colors. This project is particularly useful for applications such as image restoration, enhancing black-and-white photos, and other tasks where colorization is required.

### Key Features

- **Autoencoder Architecture**: The model uses a symmetrical encoder-decoder architecture to capture the essential features of grayscale images and reconstruct the corresponding color images.
- **Self-Supervised Learning**: The model is trained without explicit color labels, making it adaptable to various datasets.
- **Custom Color Loss**: A specialized color loss function ensures that the color channels are well-distributed and that the generated images have vibrant and accurate colors.
- **Perceptual Loss**: Utilizes a pre-trained network to capture high-level features, ensuring the generated colors align with perceptual reality.
- **Image Augmentation**: Data augmentation techniques are used to improve the robustness of the model.
- **Flexible Input Handling**: The model accepts grayscale images as input and outputs fully colorized images.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Grayscale-to-Color-Autoencoder.git
# Model Weights and Notebook

- **Model Weights**: The trained model weights are saved in the file `model_weights.h5`. You can load these weights to replicate the results or further fine-tune the model.
  
- **Jupyter Notebook**: The file `Autoencoder.ipynb` contains the full training pipeline, including data preprocessing, model architecture, and training code. You can use this notebook to understand the model's workflow and experiment with different parameters.
- **Model Dataset**: The Dataset for training the model can be found in the  `Train` folder.

# Model Architecture

The model consists of the following layers:

*   **Encoder**: Multiple convolutional layers with downsampling to capture essential features from the grayscale input.
*   **Decoder**: Up-sampling layers combined with skip connections to reconstruct the color image.
*   **Loss Functions**:
    *   **Reconstruction Loss**: Mean Squared Error (MSE) between the true and predicted images.
    *   **Color Loss**: A custom loss function designed to maintain color balance and vibrancy and single color dominance.
    *   **Perceptual Loss**: Utilizes a pre-trained network to capture high-level features, ensuring the generated colors align with perceptual reality.
