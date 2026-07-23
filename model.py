#Importing required libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



#Loading the dataset
df = pd.read_csv("emotions.csv")
print(df.head())
print()


#Dropping rows with Null Values
df.dropna(inplace=True)

#Seperating features and labels
x = df['text']
y  = df['label']

#Splitting dataset for test and train
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#Apllying TFidVectorizer
vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,3),max_features=30000,min_df=2,sublinear_tf=True)
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

#Training the model using LogisticRegression
model = LogisticRegression(max_iter=1000,C=3,class_weight='balanced')
model.fit(x_train,y_train)

#Predictig 
y_pred = model.predict(x_test)
#print(y_pred)
print()

#Accuracy Score
print("Accuracy Score:",accuracy_score(y_pred,y_test))
print()

#Gving a simple text to detect emotion
sample = [""]
sample = vectorizer.transform(sample)
print("I am really disappointed by the product. It’s frustrating -->",model.predict(sample))
print()

#Classification report
print("Classification Report:")
print(classification_report(y_test,y_pred))
print()

#Confusin Matrix
cm = confusion_matrix(y_test,y_pred)
labels = ['sadness','joy','love','anger','fear','surprise']

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()