# C242-Fs01 Machine Learning Documentation

This repository contains Jupyter notebooks designed for preprocessing, training, and prediction workflows using machine learning models. Below is an overview of each file and its functionalities.

---

## Files Overview

### 1. **preprocessing.ipynb**
#### Description:
This notebook handles the preprocessing of raw data, preparing it for model training. Key operations include cleaning, transforming, and splitting the dataset.

#### Key Features:
- **Data Cleaning:** Removes null values, handles missing data, and standardizes formats.
- **Feature Engineering:** Includes encoding categorical variables, scaling numerical features, and creating derived features.
- **Train-Test Split:** Divides the dataset into training and testing subsets.
- **Visualization:** Provides plots and graphs to understand the data distribution.

#### Usage:
1. Open the notebook in Jupyter or any compatible environment.
2. Update the input data path.
3. Run the cells sequentially to generate preprocessed datasets.

---

### 2. **train_model.ipynb**
#### Description:
This notebook is used to train a machine learning model using the preprocessed data. It includes model definition, training, and evaluation.

#### Key Features:
- **Model Architecture:** Defines the architecture using Resnet50 as encoder and U-Net as decoder.
- **Hyperparameter Tuning:** Includes adjustable parameters for optimizing model performance.
- **Training Pipeline:** Executes the training loop and tracks metrics like loss, accuracy, jaccard coefficient, dice loss.
- **Model Evaluation:** Evaluates the model on the test dataset and generates performance metrics such as pixel accuracy, jaccard coefficient and IoU.

#### Usage:
1. Ensure that the preprocessed dataset from `preprocessing.ipynb` is available.
2. Open the notebook and configure the training parameters.
3. Run the cells to train and save the model.

---

### 3. **predict.ipynb**
#### Description:
This notebook uses the trained model to make predictions on new data.

#### Key Features:
- **Model Loading:** Loads the trained model from a specified directory.
- **Data Input:** Accepts new data in the required format.
- **Prediction Pipeline:** Processes the input data and outputs predictions.
- **Visualization:** Displays the prediction results in a user-friendly format (e.g., visuals).

#### Usage:
1. Ensure the trained model file is accessible.
2. Provide the input data for predictions.
3. Run the cells to generate predictions and visualize the results.

---

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Bangkit-Capstone-Solafune-C242-FS01/machine-learning
   cd machine-learning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Lab:
   ```bash
   jupyter lab
   ```
4. Navigate to the desired notebook and follow the instructions in the file.

---

## Requirements
- Python 3.9+
- jupyter notebook
- Common Python libraries: NumPy, Pandas, Scikit-learn, Matplotlib, TensorFlow/PyTorch (adjust based on model framework)
- rasterio
- albumentations
- timm
- segmentation-models

## Preprocessing Requirements
- OpenCV
- Patchify
- Scikit-image

---

