import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Reddit API Authentication
reddit = praw.Reddit(
    client_id="eM-eMP4l-1s63e1B1376wA",
    client_secret="-nzNaYKAcKj0czeqixJ_dJB3ms0elA",
    user_agent="python:RedditScraper:1.0 (by u/bonginn)"
)

# Choose a subreddit and collect posts
subreddits = ["ArtificialInteligence", "StockMarket", "Music", "gaming", "business", "Scams", "jobs", 'community', 'nba', 'movies', 'education', 'comicbooks', 'Meditation']
posts = []

# Scrape posts
for sub in subreddits:
    cnt = 0
    subreddit = reddit.subreddit(sub)
    for post in subreddit.hot(limit=5000):  # Adjust limit as needed
        cnt += 1
        posts.append([post.title, post.selftext, post.score, post.num_comments, post.created_utc])

    print(f"Subreddit: {sub}, Posts: {cnt}")
    
# Convert to DataFrame
df = pd.DataFrame(posts, columns=["Title", "Body", "Score", "Comments", "Timestamp"])
df.to_csv("reddit_data.csv", index=False)
print("Scraping complete. Data saved!")

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score["compound"]  # -1 (negative) to +1 (positive)

# Apply sentiment analysis
df["Sentiment"] = df["Body"].apply(get_sentiment)
df["Sentiment_Label"] = df["Sentiment"].apply(lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")
df.to_csv("reddit_sentiment.csv", index=False)
print("Sentiment analysis complete. Data saved!")

