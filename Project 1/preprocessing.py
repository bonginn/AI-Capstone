import pandas as pd

def data_preprocessing(df):
    """
    Preprocess the data by removing rows with missing body, 
    keeping only the body and sentiment columns, and transforming the sentiment labels to integers.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    Returns
    -------
    df : pandas.DataFrame
        The preprocessed DataFrame.
    """
    # Delete rows with missing body
    df = df.dropna(subset=['Body'])

    # Keep only the body and sentiment columns
    df = df[['Body', 'Sentiment_Label']]

    # Trnasform the sentiment labels to integers
    sentiment_dict = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['Sentiment_Label'] = df['Sentiment_Label'].map(sentiment_dict)

    return df