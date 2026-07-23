# Simple-Emotion-Prediction-Model
# Emotion Detection using Machine Learning

## 📌 Overview
This project is a text-based Emotion Detection system built using Machine Learning. It classifies user text into one of six emotions using Natural Language Processing (NLP) techniques and Logistic Regression.

The model processes text using TF-IDF Vectorization and predicts emotions with high accuracy.

## 🚀 Features
- Text preprocessing using TF-IDF
- Emotion classification using Logistic Regression
- Supports six emotions:
  - 😊 Joy
  - 😢 Sadness
  - ❤️ Love
  - 😠 Anger
  - 😨 Fear
  - 😲 Surprise
- Generates classification report
- Displays confusion matrix for model evaluation

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 📂 Dataset
The project uses the `emotions.csv` dataset containing:
- Text
- Emotion Label

## ⚙️ Machine Learning Pipeline
1. Load the dataset
2. Remove missing values
3. Split into training and testing sets
4. Convert text into TF-IDF vectors
5. Train a Logistic Regression model
6. Predict emotions
7. Evaluate using:
   - Accuracy Score
   - Classification Report
   - Confusion Matrix

## 📊 Model Performance
The model achieves approximately **90% accuracy** on the test dataset.

## 📷 Output
The project displays:
- Predicted emotion for sample text
- Accuracy score
- Classification report
- Confusion matrix visualization

## ▶️ How to Run
1. Clone this repository
2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. Place `emotions.csv` in the project folder.
4. Run:

```bash
python model.py
```

## 📚 Future Improvements
- Deploy using Flask or Streamlit
- Real-time emotion detection
- Deep Learning models (LSTM/BERT)
- Web interface for user interaction

## 👨‍💻 Author
Yasin Rafeeque
