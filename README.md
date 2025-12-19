# Privacy-Preserving Diabetes Mellitus Prediction using Machine Learning and Homomorphic Encryption

## Title

Privacy-Preserving Diabetes Mellitus Prediction using Machine Learning and Homomorphic Encryption

## Description

This project presents a privacy-preserving framework for predicting diabetes mellitus using machine learning (ML) integrated with homomorphic encryption (HE). The key objective is to enable accurate diabetes risk prediction while ensuring that sensitive patient data remains encrypted throughout the prediction process. By combining conventional ML models with HE, the system allows computations to be performed directly on encrypted data, addressing privacy and security concerns in healthcare applications.

The project is designed for educational and research purposes, demonstrating how privacy-enhancing technologies can be integrated into medical AI pipelines.

## Dataset Information

The dataset used in this project is the **Pima Indians Diabetes Dataset**, a widely used benchmark dataset for diabetes mellitus prediction tasks.

* Dataset type: Structured tabular clinical dataset
* Population: Pima Indian women aged 21 years and above
* Number of instances: 768 samples
* Number of features: 8 clinical attributes
* Target variable: Diabetes status (binary classification: diabetic / non-diabetic)

### Features

The dataset includes the following attributes:

* Number of pregnancies
* Plasma glucose concentration
* Diastolic blood pressure (mm Hg)
* Triceps skin fold thickness (mm)
* Serum insulin (mu U/ml)
* Body mass index (BMI)
* Diabetes pedigree function
* Age (years)

### Data Source

The dataset is publicly available and commonly accessed via the UCI Machine Learning Repository.

### Preprocessing

Preprocessing steps applied in this project include:

* Handling missing or zero-valued entries
* Feature normalization / standardization
* Train–test split for model evaluation

*The dataset contains no personally identifiable information and is suitable for research and educational use when handled responsibly.**

## Code Information

The codebase is organized to clearly separate data processing, model training, evaluation, and encryption-related operations.

Typical components include:

* Data preprocessing scripts
* Machine learning model training (e.g., Logistic Regression, SVM, Random Forest, or MLP)
* Model evaluation (accuracy, precision, recall, ROC-AUC)
* Homomorphic encryption setup and encrypted inference

The implementation demonstrates how a trained ML model can be used to perform predictions on encrypted input data without decrypting it during computation.

## Usage Instructions

1. Clone or download this repository.
2. Prepare the dataset by placing it in the designated data directory.
3. Run the preprocessing script to clean and normalize the data.
4. Train the machine learning model using the provided training script.
5. Initialize the homomorphic encryption context.
6. Perform encrypted inference on test samples.
7. Evaluate and compare performance between plaintext and encrypted predictions.

Example (simplified):

```bash
python preprocess.py
python train_model.py
python encrypted_inference.py
```

## Requirements

The project is implemented in Python. Required dependencies typically include:

* Python >= 3.8
* numpy
* pandas
* scikit-learn
* matplotlib / seaborn (for visualization)
* Homomorphic encryption library (e.g., Pyfhel, TenSEAL, or equivalent)

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Methodology

1. **Data Preprocessing**: Clean the dataset, handle missing values, normalize features, and split data into training and testing sets.
2. **Model Training**: Train a supervised ML classifier using plaintext data to achieve optimal predictive performance.
3. **Model Evaluation**: Evaluate the trained model using standard performance metrics.
4. **Homomorphic Encryption Setup**: Initialize the HE scheme and generate encryption keys.
5. **Encrypted Inference**: Encrypt input features and perform model inference directly on encrypted data.
6. **Result Comparison**: Compare encrypted prediction results with plaintext predictions to assess accuracy and computational overhead.

## Citations

Pima Indians Diabetes Database [Data set]. Originally from the National Institute of Diabetes and Digestive and Kidney Diseases. Donor: Vincent Sigillito, The Johns Hopkins University, Applied Physics Laboratory. Available from the UCI Machine Learning Repository.

## Contribution Guidelines

Contributions are welcome. Users may:

* Report issues or bugs
* Suggest improvements
* Submit pull requests for enhancements

Please ensure that contributions follow ethical guidelines, especially when handling medical data.
