# Task5
🧠 AIML – Task 5: Decision Trees & Random Forests

This project is part of the Artificial Intelligence & Machine Learning (AIML) coursework.
The task focuses on building Decision Tree and Random Forest models for classification using Python.

📌 Objective

Understand how tree-based ML models work

Train and evaluate a Decision Tree Classifier

Visualize a decision tree

Control tree depth to avoid overfitting

Train a Random Forest Classifier and compare performance

Interpret feature importances

Evaluate using metrics and cross-validation

📂 Dataset

You can use any classification dataset.

Example used: Heart Disease Dataset (CSV).
Make sure your CSV includes the target/output column (usually named target).

📎 Place your dataset in the same folder and update only this line in code:

df = pd.read_csv("your_dataset.csv")

🛠️ Tools & Libraries

Python

Scikit-learn

Pandas

NumPy

Matplotlib (optional for visualizing trees)

Graphviz (optional for tree visualization)

Install dependencies:

pip install pandas numpy scikit-learn matplotlib

▶️ How to Run the Code

Place your dataset CSV in the project folder

Update the dataset filename in the script (if needed)

Run the script:

python task5.py


The console will display:

Training accuracy

Confusion matrix

Classification report

Random Forest accuracy

Feature importances

📊 Outputs Example

Model accuracy

Confusion Matrix

Classification Report

Tree depth & overfitting analysis

Random Forest comparison

Feature importance ranking

📁 Project Structure
📦 AIML Task 5
 ┣ 📄 task5.py
 ┣ 📄 dataset.csv
 ┗ 📄 README.md

✨ Key Learnings

How Decision Trees split data based on Information Gain / Gini

How max_depth & min_samples_split prevent overfitting

Why Random Forests outperform a single tree

How to evaluate models properly

👩‍💻 Author

G Harshitha
student of aiml cse 
