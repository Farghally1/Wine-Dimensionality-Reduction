# Wine Dimensionality Reduction and Deep Learning Classification

## Project Overview

This project investigates the effectiveness of **Autoencoder-based Dimensionality Reduction** and evaluates its impact on Deep Learning classification performance.

Using the Wine Dataset from Scikit-learn, an Autoencoder Neural Network is developed to compress the original 13 features into a 5-dimensional latent representation.

Two Deep Learning classification models are then implemented and compared:

- **Encoded Features Model:** Uses the 5-dimensional representation extracted by the trained Autoencoder.
- **Original Features Model:** Uses all 13 original features.

The main objective is to determine whether a compressed feature representation can preserve relevant information and achieve competitive classification performance while reducing input dimensionality.

## Project Objectives

- Perform Exploratory Data Analysis (EDA).
- Analyze statistical properties and feature distributions.
- Investigate correlations between numerical features.
- Apply data preprocessing and feature scaling.
- Implement an Encoder-Decoder Neural Network.
- Extract compressed latent representations.
- Build and train Deep Learning classification models.
- Evaluate and compare classification performance.
- Visualize latent representations using PCA.

## Dataset

**Dataset:** Wine Dataset  
**Source:** Scikit-learn (`sklearn.datasets.load_wine`)

| Property | Value |
|---|---|
| Total Samples | 178 |
| Original Features | 13 |
| Target Classes | 3 |
| Problem Type | Multi-class Classification |

The dataset contains chemical analysis measurements of wines belonging to three different classes.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Conduct an initial investigation of the dataset, including:

- Statistical Analysis
- Feature Distribution Plots
- Box Plots and Outlier Detection
- Feature Correlation Analysis
- Correlation with Target
- Class Distribution Visualization

### 2. Data Preprocessing

Prepare the dataset for Deep Learning:

- Separate Features and Target.
- Split the dataset into Training and Testing sets (80/20).
- Apply StandardScaler.
- Prevent Data Leakage by fitting the scaler only on training data.

### 3. Autoencoder Implementation

Build an Encoder-Decoder Neural Network to learn a compressed representation of the original input features.

**Encoder Architecture**

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 13 | - |
| Dense | 64 | ReLU |
| Dense | 32 | ReLU |
| Latent Representation | 5 | ReLU |

**Decoder Architecture**

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 5 | - |
| Dense | 32 | ReLU |
| Dense | 64 | ReLU |
| Output | 13 | Linear |

**Training Configuration**

- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Epochs: 100

### 4. Encoded Feature Extraction

Use the trained Encoder to transform the original 13-dimensional feature space into a 5-dimensional latent representation.

These encoded features are subsequently used as input for the first classification model.

### 5. Deep Learning Classification

Two Neural Network classifiers are implemented to investigate the effect of dimensionality reduction.

**Model A: Encoded Features Classification**

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 5 | - |
| Dense | 32 | ReLU |
| Dense | 16 | ReLU |
| Output | 3 | Softmax |

**Model B: Original Features Classification**

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 13 | - |
| Dense | 64 | ReLU |
| Dense | 32 | ReLU |
| Dense | 16 | ReLU |
| Output | 3 | Softmax |

**Training Configuration**

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Output Activation: Softmax

### 6. Model Evaluation and Comparison

Evaluate both classification models using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Macro averaging is used for Precision, Recall, and F1-Score to give equal importance to all target classes.

### 7. Visualization

Generate visualizations to analyze model behavior and feature representations:

- Autoencoder Training and Validation Loss.
- Classification Training and Validation Loss.
- Confusion Matrices for both classifiers.
- PCA Visualization of the Encoded Features.

## Model Comparison

The project compares two approaches:

| Aspect | Encoded Model | Original Model |
|---|---|---|
| Input Features | 5 | 13 |
| Architecture | 32 → 16 → 3 | 64 → 32 → 16 → 3 |
| Output Activation | Softmax | Softmax |
| Loss Function | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy |

The comparison focuses on determining whether the Autoencoder can reduce dimensionality while maintaining useful classification performance.

Actual experimental results will be added after completing model training and evaluation.

## Project Structure

```text
Wine-Dimensionality-Reduction/
│
├── notebooks/
│   └── wine_autoencoder.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Farghally1/Wine-Dimensionality-Reduction.git
```

### 2. Navigate to the Project Directory

```bash
cd Wine-Dimensionality-Reduction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook and execute the cells sequentially.

## Future Improvements

- Compare Autoencoder performance with PCA-based dimensionality reduction.
- Experiment with different latent dimensions.
- Evaluate additional classification algorithms.
- Apply hyperparameter tuning.
- Investigate the effect of dimensionality reduction on model generalization.

## Author

**Mohammed Farghally Abd El Fattah**

Artificial Intelligence Student

GitHub: [Farghally1](https://github.com/Farghally1)

---

This project was developed for educational and practical exploration of dimensionality reduction, representation learning, and Deep Learning classification.
