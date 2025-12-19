# Privacy-Preserving Diabetes Mellitus Prediction using Machine Learning and Homomorphic Encryption

## Project Overview
This project presents a privacy-preserving framework for predicting diabetes mellitus using machine learning (ML) integrated with **Homomorphic Encryption (HE)**. The key objective is to balance high-accuracy disease prediction with patient data privacy.

The system utilizes **Paillier Encryption**, a partial homomorphic encryption scheme, to secure sensitive patient data. It demonstrates a workflow where clinical data can be encrypted for storage and transmission, ensuring that sensitive attributes remain protected outside the trusted training environment.

## Dataset Information
The project uses the **Pima Indians Diabetes Dataset**, a widely used benchmark dataset for diabetes prediction tasks.

* **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (accessed via UCI Machine Learning Repository).
* **Instances:** 768 samples
* **Target:** Binary classification (Diabetic / Non-Diabetic)

### Features
1.  **Pregnancies**: Number of times pregnant
2.  **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3.  **BloodPressure**: Diastolic blood pressure (mm Hg)
4.  **SkinThickness**: Triceps skin fold thickness (mm)
5.  **Insulin**: 2-Hour serum insulin (mu U/ml)
6.  **BMI**: Body mass index (weight in kg/(height in m)^2)
7.  **DiabetesPedigreeFunction**: Diabetes pedigree function
8.  **Age**: Age (years)

## Repository Structure

The codebase is modularized into three sequential steps to separate concerns between data science and security operations:

* **`1_data_preprocessing.py`**:
    * Handles missing values (zero-replacement strategy) and outlier removal.
    * Generates 5 different resampling variations (RUS, ROS, SMOTE, Tomek, NearMiss) to address class imbalance.
    * **Output:** Generates `resampling_visualization.png` (PCA & Class Balance charts) and `processed_datasets.pkl`.
 
* **`2_model_training.py`**:
    * Trains multiple ML models (XGBoost, CatBoost, Random Forest, SVM, etc.).
    * Uses `RandomizedSearchCV` for hyperparameter optimization across all models.
    * **Output:** Generates `model_performance_comparison.png`, saves the best performing model as `best_model.pkl`, and creates `model_results.csv`.

* **`3_homomorphic_encryption.py`**:
    * Implements the **Paillier Cryptosystem** using the `phe` library.
    * Demonstrates key generation (Public/Private), data encryption, and decryption verification.
    * Simulates a secure pipeline where the model is tested against decrypted data to verify integrity.

## Installation & Requirements

The project is implemented in Python (>= 3.8). Install the required dependencies using the command below:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost catboost imbalanced-learn phe
```

**Key Libraries:**
* `scikit-learn` & `imbalanced-learn`: For standard ML pipelines and resampling.
* `phe`: Python Paillier library for Homomorphic Encryption.
* `xgboost` & `catboost`: For high-performance gradient boosting algorithms.

## Usage Instructions

Run the scripts in the following numerical order to reproduce the full pipeline:

**Step 1: Preprocessing & Visualization**
Clean the data and visualize how different resampling techniques affect the data distribution.
```bash
python 1_data_preprocessing.py
```

**Step 2: Model Training & Selection**
Train all candidate models, perform hyperparameter tuning, and identify the best performer.
```bash
python 2_model_training.py
```

**Step 3: Encrypted Pipeline**
Generate encryption keys, encrypt the dataset, and verify the model's performance on the secure pipeline.
```bash
python 3_homomorphic_encryption.py
```

## Methodology

1.  **Data Preprocessing**: Handling zero-values in glucose/insulin columns and removing outliers based on the 95th-99th percentiles.
2.  **Resampling**: Comparative analysis of Over-sampling (SMOTE, ROS) and Under-sampling (RUS, NearMiss, Tomek) techniques.
3.  **Model Selection**: Rigorous evaluation of tree-based models (Random Forest, XGBoost) and linear models (Logistic Regression) using Randomized Grid Search.
4.  **Privacy Preservation**: Implementation of the **Paillier Cryptosystem**, allowing the dataset to be encrypted using a public key. The private key is required to decrypt the data for the inference/training phase, preventing unauthorized access during storage.

## Citations
* *Pima Indians Diabetes Database* [Data set]. Originally from the National Institute of Diabetes and Digestive and Kidney Diseases. Donor: Vincent Sigillito, The Johns Hopkins University, Applied Physics Laboratory.
* *Paillier, P. (1999)*. Public-Key Cryptosystems Based on Composite Degree Residuosity Classes.

## Contribution
Contributions are welcome! Please open an issue or submit a pull request for improvements. Users are reminded to follow ethical guidelines when handling medical data.
