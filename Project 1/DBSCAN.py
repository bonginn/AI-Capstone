# import libraries
from sklearn.metrics import adjusted_rand_score, silhouette_score,
from sklearn.metrics import normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from preprocessing import  data_preprocessing
from sklearn.cluster import DBSCAN
import numpy as np

# Read Data from CSV
df = pd.read_csv('reddit_sentiment.csv')

# Preprocess the data
df = data_preprocessing(df)
X = df['Body']
y = df['Sentiment_Label']

# Set the number of clusters
num_clusters = 3

# Experiment 1 - Vectorizer

# Vectorize the data using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10)
X_tfidf = vectorizer.fit_transform(X)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_tfidf)

# Evaluate the clustering 
ari = adjusted_rand_score(y, dbscan.labels_)
silhouette = silhouette_score(X_tfidf, dbscan.labels_)
nmi = normalized_mutual_info_score(y, dbscan.labels_)

# Print the results
print(f'Adjusted Rand Index: {ari}')
print(f'Silhouette Score: {silhouette}')
print(f'Normalized Mutual Information: {nmi}')

# Vectorize the data using CountVectorizer
vectorizer = CountVectorizer(max_features=10)
X_count = vectorizer.fit_transform(X)
dbscan_count = DBSCAN(eps=0.5, min_samples=5)
dbscan_count.fit(X_count)

# Evaluate the clustering
ari = adjusted_rand_score(y, dbscan_count.labels_)
silhouette = silhouette_score(X_count, dbscan_count.labels_)
nmi = normalized_mutual_info_score(y, dbscan_count.labels_)

# Print the results
print(f'Adjusted Rand Index: {ari}')
print(f'Silhouette Score: {silhouette}')
print(f'Normalized Mutual Information: {nmi}')

# import word2vec model
import gensim.downloader as api
w2v_model = api.load("word2vec-google-news-300")  

# Function to convert text to word2vec
def text_to_w2v(text, model=w2v_model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(300)  

# Convert the text data to word2vec
X_w2v = np.array([text_to_w2v(text) for text in X])

# Build the DBSCAN model and fit it
dbscan_w2v = DBSCAN(eps=0.5, min_samples=5)
dbscan_w2v.fit(X_w2v)

# Evaluate the clustering
ari = adjusted_rand_score(y, dbscan_w2v.labels_)
silhouette = silhouette_score(X_w2v, dbscan_w2v.labels_)
nmi = normalized_mutual_info_score(y, dbscan_w2v.labels_)

# Print the results
print(f'Adjusted Rand Index: {ari}')
print(f'Silhouette Score: {silhouette}')
print(f'Normalized Mutual Information: {nmi}')

# Experiment 2 - Over-sampling, SMOTE and Under-sampling
# import libraries
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_w2v, y)

# Build the DBSCAN model and fit it
model_ros = DBSCAN(eps=0.5, min_samples=5)
model_ros.fit(X_ros)

# Evaluate the clustering
ari = adjusted_rand_score(y_ros, model_ros.labels_)
silhouette = silhouette_score(X_ros, model_ros.labels_)
nmi = normalized_mutual_info_score(y_ros, model_ros.labels_)

# Print the results
print(f'Adjusted Rand Index: {ari}')
print(f'Silhouette Score: {silhouette}')
print(f'Normalized Mutual Information: {nmi}')

# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_w2v, y)

# Build the DBSCAN model and fit it
model_smote = DBSCAN(eps=0.5, min_samples=5)
model_smote.fit(X_smote)

# Evaluate the clustering
ari = adjusted_rand_score(y_smote, model_smote.labels_)
silhouette = silhouette_score(X_smote, model_smote.labels_)
nmi = normalized_mutual_info_score(y_smote, model_smote.labels_)

# Print the results
print(f'Adjusted Rand Index: {ari}')
print(f'Silhouette Score: {silhouette}')
print(f'Normalized Mutual Information: {nmi}')

# RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_w2v, y)

# Build the DBSCAN model and fit it
model_rus = DBSCAN(eps=0.5, min_samples=5)
model_rus.fit(X_rus)

# Evaluate the clustering
ari = adjusted_rand_score(y_rus, model_rus.labels_)
silhouette = silhouette_score(X_rus, model_rus.labels_)
nmi = normalized_mutual_info_score(y_rus, model_rus.labels_)

# Print the results
print(f'Adjusted Rand Index: {ari}')
print(f'Silhouette Score: {silhouette}')
print(f'Normalized Mutual Information: {nmi}')