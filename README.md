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

Next, you need to download the phishing_email.csv dataset from Kaggle or upload it into the project directory. Ensure that the dataset is formatted correctly with each email having features such as the body text, URL count, and a label indicating whether the email is phishing (1) or legitimate (0).

Once the dataset is in place, you can run the script to start the phishing email detection process:
