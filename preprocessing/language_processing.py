import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import regexp_tokenize
import preprocessor as p
from util import red, cyan


def set_up_stop_words() -> set:
    """Download and return English stop words."""
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


def remove_stop_words(tokens: list[str], stop_words: set) -> list[str]:
    """Remove stop words from token list."""
    return [t for t in tokens if t.lower() not in stop_words]


def preprocess_twitter_text(text: str) -> str:
    """Clean tweet-specific artifacts from the text."""
    return p.clean(text)


def sentiment_tokenizer(text: str) -> list[str]:
    """Tokenize input using a custom regex pattern."""
    pattern = r"""(?x)(?:[A-Z]\.)+|\w+(?:[-']\w+)*|\$?\d+(?:\.\d+)?%?|\.\.\.|[][.,;"'?():-_`]"""
    return regexp_tokenize(text, pattern)


def stem_text(tokens: list[str]) -> list[str]:
    """Stem tokens using PorterStemmer."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def process_text(text: str, stop_words: set, tweet: bool = True, verbose: bool = False) -> list[str] | None:
    """Full pipeline for preprocessing a single text."""
    try:
        if tweet:
            text = preprocess_twitter_text(text)
        tokens = sentiment_tokenizer(text)
        filtered = remove_stop_words(tokens, stop_words)
        stemmed = stem_text(filtered)
        if verbose:
            cyan(f"Text: {text}")
            print("→ Tokenized:", tokens)
            print("→ Filtered:", filtered)
            print("→ Stemmed:", stemmed)
        return stemmed
    except Exception as e:
        red(f"Error processing text: {e}")
        return None
