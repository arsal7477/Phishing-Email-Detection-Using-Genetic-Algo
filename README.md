# Phishing Email Detection Using Genetic Algorithm and Random Forest

## Overview

This project demonstrates the use of a **Genetic Algorithm (GA)** in combination with **Random Forest** for the detection of phishing emails. Phishing attacks are one of the most widespread forms of cyberattacks, which aim to steal sensitive information such as usernames, passwords, and credit card details. Detecting phishing emails automatically is a crucial task for improving cybersecurity.

In this project:
- **Genetic Algorithm (GA)** is used for optimizing feature selection.
- **Random Forest** classifier is used to detect phishing emails.
- The model is trained on a dataset of phishing emails, evaluated with common metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

## Features

- **Genetic Algorithm**: Optimizes the feature selection to improve the classifier’s performance.
- **Random Forest Classifier**: Used to classify emails as phishing or legitimate.
- **Performance Evaluation**: The model is evaluated using accuracy, precision, recall, F1-score, and confusion matrix.
- **Visualization**: Includes visualizations such as **fitness over generations**, **confusion matrix**, and **evaluation metrics** to help assess model performance.

## Dataset

The dataset used for this project is sourced from **Kaggle**, specifically the `phishing_email.csv` file. The dataset contains a collection of emails labeled as **phishing (1)** or **legitimate (0)**, with various features extracted from the email content. 

Link to the dataset: [Kaggle - Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

## Prerequisites

To run this project, make sure you have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`
- `deap` (for genetic algorithm)
- `random` (for feature selection and mutation)

You can install the necessary packages using `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn deap

## Setup and Usage

### 1. Clone the repository
```bash
git clone https://github.com/arsal7477/Phishing-Email-Detection-Using-Genetic-Algo/blob/main/Phishing_detection.ipynb
cd phishing-email-detection
```
Next, you need to download the phishing_email.csv dataset from Kaggle or upload it into the project directory. Ensure that the dataset is formatted correctly with each email having features such as the body text, URL count, and a label indicating whether the email is phishing (1) or legitimate (0).

Once the dataset is in place, you can run the script to start the phishing email detection process:
```
python Phishing_detection.ipynb
```

Upon running the script, the program will output various performance metrics such as accuracy, precision, recall, and F1-score. Additionally, it will generate the following visualizations:

Fitness Over Generations: A plot showing how the fitness of the Genetic Algorithm evolved over generations.
Confusion Matrix: A matrix showing the true positives, false positives, true negatives, and false negatives of the classifier.
Precision, Recall, and F1-Score Over Generations: A plot showing how precision, recall, and F1-score changed across generations.
Here's an example output you might get:
```
Generation 1: Best Fitness = 0.973
Generation 2: Best Fitness = 0.972
...
Final Accuracy: 95.45%
Final Precision: 96.89%
Final Recall: 95.15%
Final F1-Score: 96.01%
```
## Methodology
The first step in the methodology is Data Preprocessing, where the dataset is cleaned and preprocessed by removing irrelevant characters and extracting relevant features. The email content is processed using TF-IDF (Term Frequency-Inverse Document Frequency), which transforms the text data into numerical data suitable for machine learning.

Next, the Feature Selection using Genetic Algorithm is performed. A Genetic Algorithm (GA) is used to select the most relevant features that help distinguish phishing emails from legitimate ones. The GA iterates through several generations, selecting, crossing over, and mutating feature sets, evaluating them based on their classification performance.

Once the optimal features are selected by the GA, Model Training takes place. A Random Forest Classifier is trained using the selected features. Random Forest is chosen because of its ability to handle a large number of features and its robustness in detecting patterns in data.

After training the model, it is evaluated on a test set using various performance metrics, including accuracy, precision, recall, and F1-score. The performance is visualized over generations to assess how the Genetic Algorithm improves the classification process.

## Visualizations
The project provides several visualizations to assess model performance:

Fitness Over Generations: A plot showing how the GA's best fitness improved over generations, indicating the convergence of the optimization process.
Confusion Matrix: A matrix that displays the true positives, false positives, true negatives, and false negatives, offering a detailed view of the classifier's performance.
Precision, Recall, and F1-Score Over Generations: A plot that tracks how the model's precision, recall, and F1-score evolved during training, helping evaluate how well the model balances detecting phishing emails and avoiding false positives.

## Conclusion
In this project, we successfully demonstrated the use of a Genetic Algorithm for feature selection and Random Forest for phishing email detection. The Genetic Algorithm effectively optimized the feature set, improving the model's ability to classify phishing emails. The final model achieved an accuracy of 95.45%, precision of 96.89%, recall of 95.15%, and an F1-score of 96.01%. The visualizations provided valuable insights into how the model improved over generations, helping track the progress of the optimization process.

##Future Work
Several areas of improvement and future work could be considered:

Parameter Tuning: Fine-tuning the parameters of the Genetic Algorithm and experimenting with different classifiers such as Support Vector Machines (SVM) or Logistic Regression could further improve performance.
Larger Dataset: Testing the model on a larger dataset would help improve its generalization and enhance performance.
Real-Time Detection: Implementing the model for real-time phishing detection in email systems would make the solution practical and usable in real-world scenarios.
csharp
Copy code

