# import libraries
import pandas as pd
from preprocessing import data_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read Data from CSV
df = pd.read_csv('reddit_sentiment.csv')

# Show the Data Distribution
sentiment_distribution = df['Sentiment_Label'].value_counts()
sentiment_distribution.plot(kind='bar')
plt.xlabel('Sentiment')
plt.ylabel('Number of samples')
plt.title('Sentiment distribution')
plt.xticks(rotation=0)
plt.show()

# Preprocess the data
df = data_preprocessing(df)
print(df.head())

# Show the Data Distribution after Preprocessing
sentiment_distribution = df['Sentiment_Label'].value_counts()
sentiment_distribution.plot(kind='bar')
plt.xlabel('Sentiment')
plt.ylabel('Number of samples')
plt.title('Sentiment distribution')
plt.xticks(rotation=0)
plt.show()

X = df['Body']
y = df['Sentiment_Label']

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    df['Body'], df['Sentiment_Label'], test_size=0.2, random_state=42, 
    stratify=df['Sentiment_Label'], shuffle=True
)


# Experiment 1 - Vectorizer

# Vectorize the data using CountVectorizer
vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 2))
X_count = vectorizer.fit_transform(X)
X_train_count = vectorizer.transform(X_train)
X_test_count = vectorizer.transform(X_test)

# Cross validation for CountVectorizer
model = SVC(kernel='linear', C = 1.0)
scorings = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X_count, y, cv=skf, scoring=scorings)

# Print the results
print('Accuracy:', np.mean(scores['test_accuracy']))
print('Precision:', np.mean(scores['test_precision_macro']))
print('Recall:', np.mean(scores['test_recall_macro']))
print('F1:', np.mean(scores['test_f1_macro']))

# Draw the confusion matrix

model = SVC(kernel='linear', C=1)
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
model_tfidf = SVC(kernel='linear', C = 1.0)
scorings = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model_tfidf, X_tfidf, y, cv=skf, scoring=scorings)

# Print the results
print('Accuracy:', np.mean(scores['test_accuracy']))
print('Precision:', np.mean(scores['test_precision_macro']))
print('Recall:', np.mean(scores['test_recall_macro']))
print('F1:', np.mean(scores['test_f1_macro']))

# import word2vec model
import gensim.downloader as api
w2v_model = api.load("word2vec-google-news-300")  

# Function to convert text to word2vec
def text_to_w2v(text, model=w2v_model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(300)  # 300 維向量

# Convert text to word2vec
X_w2v = np.array([text_to_w2v(text) for text in X])
X_train_w2v = np.array([text_to_w2v(text) for text in X_train])
X_test_w2v = np.array([text_to_w2v(text) for text in X_test])

# Cross validation for word2vec
model_w2v = SVC(kernel='linear', C = 1.0)
scorings = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model_w2v, X_w2v, y, cv=skf, scoring=scorings)

# Print the results
print('Accuracy:', np.mean(scores['test_accuracy']))
print('Precision:', np.mean(scores['test_precision_macro']))
print('Recall:', np.mean(scores['test_recall_macro']))
print('F1:', np.mean(scores['test_f1_macro']))

# Confusion matrix for w2v
model_w2v = SVC(kernel='linear', C=1)
model_w2v.fit(X_train_w2v, y_train)
y_pred_w2v = model_w2v.predict(X_test_w2v)
conf_matrix = confusion_matrix(y_test, y_pred_w2v)
print("Confusion Matrix:\n", conf_matrix)
labels = ['Negative', 'Neutral', 'Positive']
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# Experiment 2 - RandomOverSampler, SMOTE, RandomUnderSampler
# import libraries
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# RandomOverSampler
# Note that we need to use the pipeline to avoid data leakage
model_ros = SVC(kernel='linear', C=1.0)
ros = RandomOverSampler(random_state=42, sampling_strategy={-1: 2500})
pipeline = Pipeline([
    ('oversample', ros),
    ('model', model_ros)
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
model_smote = SVC(kernel='linear', C=1.0)
smote = SMOTE(random_state=42)
pipeline = Pipeline([
    ('smote', smote),
    ('model', model_smote)
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
model_rus = SVC(kernel='linear', C=1.0)
rus = RandomUnderSampler(random_state=42, sampling_strategy={1: 1800})
pipeline = Pipeline([
    ('undersample', rus),
    ('model', model_rus)
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

# Experiment 3 - Data Augmentation
# import libraries
import nltk
from nltk.corpus import wordnet
import random
nltk.download('wordnet')

# Function to replace words with synonyms
def synonym_replacement(text, n=2):
    if len(text) == 0: 
        return text
    words = text.split()
    if(len(words) == 0):
        return text
    new_words = words.copy()
    for _ in range(n):
        word_idx = random.randint(0, len(words)-1)
        synonyms = wordnet.synsets(words[word_idx])
        if synonyms:
            LEN = len(synonyms)
            new_word = synonyms[random.randint(0, LEN-1)].lemmas()[0].name()
            new_words[word_idx] = new_word
    return " ".join(new_words)

# Replace words with synonyms and vectorize the data using TfidfVectorizer
X_train_augmented = [synonym_replacement(text, n=50) for text in X_train]
X_train_augmented_tfidf = vectorizer.transform(X_train_augmented)
X_train_augmented_tfidf_smote, y_train_smote = smote.fit_resample(X_train_augmented_tfidf, y_train)
# Build the model and train it
model_augmented_smote = SVC(kernel='linear', C=1.0)
model_augmented_smote.fit(X_train_augmented_tfidf_smote, y_train_smote)

# Test the model
y_pred = model_augmented_smote.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Augmented Accuracy: {accuracy:.2f}')
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

print(classification_report(y_test, y_pred, target_names=labels))