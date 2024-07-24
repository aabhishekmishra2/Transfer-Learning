# Advanced Avian Species Differentiation Utilizing Transfer Learning Techniques

## Project Overview

This project demonstrates the use of transfer learning to classify images of ducks and chickens using a pre-trained convolutional neural network (CNN). The project is implemented in a Google Colab notebook and covers the following steps:

1. **Data Collection**: Downloading and organizing images of ducks and chickens.
2. **Data Preprocessing**: Resizing, normalizing, and splitting the data.
3. **Model Selection**: Fine-tuning a pre-trained CNN model.
4. **Training**: Compiling and training the model.
5. **Evaluation**: Generating a classification report.
6. **Deployment**: Saving the model and implementing a prediction function.

## Getting Started

### Prerequisites

- Google Colab account
- Use Kaggle or Colab to run this code with enabling GPU
- Basic knowledge of Python and deep learning
- Libraries: TensorFlow, Pytorch Keras, NumPy, Pandas, Matplotlib

### Data Collection

1. Download approximately 350 images each of ducks and chickens from the internet.
2. Organize the images into two directories: `chicken-images` and `duck-images`.

### Running the Notebook

1. Open the provided Google Colab notebook.
2. Upload the collected images to your Google Drive and mount the drive in the Colab notebook.
3. Run the cells step-by-step, following the instructions and explanations provided.

### Results

The final output of the notebook is a classification report that evaluates the performance of the fine-tuned model, including precision, recall, and F1-score.

![download](https://github.com/user-attachments/assets/1c50d1d7-19d9-4405-a746-1075157de274)

## Project Structure
- **data/**: This directory contains the image data for the project.
  - **duck-images/**: Contains images of ducks.
  - **chicken-images/**: Contains images of chickens.
- **Code.py**: The main python file that includes the data preprocessing, model training, and evaluation.

