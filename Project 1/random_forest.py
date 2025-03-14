# import libraries
import pandas as pd
from preprocessing import data_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read Data from CSV
df = pd.read_csv('reddit_sentiment.csv')

# Preprocess the data
df = data_preprocessing(df)
X = df['Body']
y = df['Sentiment_Label']

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    df['Body'], df['Sentiment_Label'], test_size=0.2, random_state=42, 
    stratify=df['Sentiment_Label'], shuffle=True
)

# Experiment 1 - veoctorizer

# Vectorize the data using CountVectorizer
vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 2))
X_count = vectorizer.fit_transform(X)
X_train_count = vectorizer.transform(X_train)
X_test_count = vectorizer.transform(X_test)

# Cross validation for CountVectorizer
model = RandomForestClassifier(n_estimators=100, random_state=42)
scorings = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X_count, y, cv=skf, scoring=scorings)

# Print the results
print('Accuracy:', np.mean(scores['test_accuracy']))
print('Precision:', np.mean(scores['test_precision_macro']))
print('Recall:', np.mean(scores['test_recall_macro']))
print('F1:', np.mean(scores['test_f1_macro']))

# Draw the confusion matrix
model.fit(X_train_count, y_train)
y_pred = model.predict(X_test_count)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
labels = ['Negative', 'Neutral', 'Positive']
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# Vectorize the data using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Cross validation for TfidfVectorizer
model_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)
scorings = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model_tfidf, X_tfidf, y, cv=skf, scoring=scorings)

# Print the results
print('Accuracy:', np.mean(scores['test_accuracy']))
print('Precision:', np.mean(scores['test_precision_macro']))
print('Recall:', np.mean(scores['test_recall_macro']))
print('F1:', np.mean(scores['test_f1_macro']))

# Draw the confusion matrix
model_tfidf.fit(X_train_tfidf, y_train)
y_pred = model_tfidf.predict(X_test_tfidf)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
labels = ['Negative', 'Neutral', 'Positive']
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# import the word2vec model
import gensim.downloader as api
w2v_model = api.load("word2vec-google-news-300")  

# Function to convert text to word2vec
def text_to_w2v(text, model=w2v_model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(300)  

# Convert text to word2vec
X_w2v = np.array([text_to_w2v(text) for text in X])
X_train_w2v = np.array([text_to_w2v(text) for text in X_train])
X_test_w2v = np.array([text_to_w2v(text) for text in X_test])


# Cross validation for word2vec
model_w2v = RandomForestClassifier(n_estimators=100, random_state=42)
scorings = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model_w2v, X_w2v, y, cv=skf, scoring=scorings)

# Print the results
print('Accuracy:', np.mean(scores['test_accuracy']))
print('Precision:', np.mean(scores['test_precision_macro']))
print('Recall:', np.mean(scores['test_recall_macro']))
print('F1:', np.mean(scores['test_f1_macro']))

# Experiment 2 - RandomOverSampler, SMOTE, RandomUnderSampler

# Import the libraries
from imblearn.over_sampling import  RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# RandomOverSampler
# Note that we need to use the pipeline to avoid data leakage
model_ros = RandomForestClassifier(n_estimators=100, random_state=42)
ros = RandomOverSampler(random_state=42)
pipeline = Pipeline([
    ('ROS', ros), 
    ('RF', model_ros)
])
scorings = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross validation for RandomOverSampler
scores = cross_validate(pipeline, X_tfidf, y, cv=cv, scoring=scorings)

# Print the results
print('Accuracy:', np.mean(scores['test_accuracy']))
print('Precision:', np.mean(scores['test_precision_macro']))
print('Recall:', np.mean(scores['test_recall_macro']))
print('F1:', np.mean(scores['test_f1_macro']))


# SMOTE
model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
smote = SMOTE(random_state=42)
pipeline = Pipeline([
    ('SMOTE', smote), 
    ('RF', model_smote)
])
scorings = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross validation for SMOTE
scores = cross_validate(pipeline, X_tfidf, y, cv=cv, scoring=scorings)

# Print the results
print('Accuracy:', np.mean(scores['test_accuracy']))
print('Precision:', np.mean(scores['test_precision_macro']))
print('Recall:', np.mean(scores['test_recall_macro']))
print('F1:', np.mean(scores['test_f1_macro']))

# RandomUnderSampler
model_rus = RandomForestClassifier(n_estimators=100, random_state=42)
rus = RandomUnderSampler(random_state=42, sampling_strategy={1: 1800})
pipeline = Pipeline([
    ('RUS', rus), 
    ('RF', model_rus)
])
scorings = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross validation for RandomUnderSampler
scores = cross_validate(pipeline, X_tfidf, y, cv=cv, scoring=scorings)

# Print the results
print('Accuracy:', np.mean(scores['test_accuracy']))
print('Precision:', np.mean(scores['test_precision_macro']))
print('Recall:', np.mean(scores['test_recall_macro']))
print('F1:', np.mean(scores['test_f1_macro']))