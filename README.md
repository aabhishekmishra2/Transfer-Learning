# Duck vs Chicken Image Classification using Transfer Learning

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
- Basic knowledge of Python and deep learning
- Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib

### Data Collection

1. Download approximately 100 images each of ducks and chickens from the internet.
2. Organize the images into two directories: `data/ducks` and `data/chickens`.

### Running the Notebook

1. Open the provided Google Colab notebook.
2. Upload the collected images to your Google Drive and mount the drive in the Colab notebook.
3. Run the cells step-by-step, following the instructions and explanations provided.

### Results

The final output of the notebook is a classification report that evaluates the performance of the fine-tuned model, including precision, recall, and F1-score.

## Project Structure
- **data/**: This directory contains the image data for the project.
  - **ducks/**: Contains images of ducks.
  - **chickens/**: Contains images of chickens.
- **notebook/**: This directory contains the Jupyter Notebook for the project.
  - **Duck_vs_Chicken_Classification.ipynb**: The main notebook that includes the data preprocessing, model training, and evaluation.

