import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import regexp_tokenize
import preprocessor as p


def remove_stop_words(tokenized_text):
    # Download stopwords if not already downloaded
    nltk.download('stopwords')
    # Get the set of English stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords from the list of words
    filtered_tokens = [token for token in tokenized_text if token.lower() not in stop_words]

    return filtered_tokens


def preprocess_twitter_text(tweet_text):
    # Clean and preprocess the tweet text
    cleaned_text = p.clean(tweet_text)
    return cleaned_text


def sentiment_tokenizer(text):
    # Define the regular expression pattern for word tokenization
    pattern = r'''(?x)          # Set flag to allow verbose regex
                  (?:[A-Z]\.)+  # Match abbreviations like U.S.A.
                  | \w+(?:[-']\w+)*   # Match words with optional hyphens or apostrophes
                  | \$?\d+(?:\.\d+)?%?  # Match currency and percentages, e.g., $10.99 or 50%
                  | \.\.\.       # Match ellipsis (...)
                  | [][.,;"'?():-_`]  # Match specific punctuation marks
               '''

    # Tokenize the input text using the regular expression pattern
    words = regexp_tokenize(text, pattern)

    return words


def stem_text(text):
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in text]
    return stemmed


def process_text(text, tweet=False):
    if tweet:
        text = preprocess_twitter_text(text)
    tokenized = sentiment_tokenizer(text)
    print("Tokenized:", tokenized)

    filtered = remove_stop_words(tokenized)
    print("Filtered:", filtered)

    stemmed = stem_text(filtered)
    print("Stemmed:", stemmed)

    return stemmed


input_text = "I love running in the mornings. Runners are running away while running shoes are running."

processed_text = process_text(input_text)
print(processed_text)


