import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import regexp_tokenize
import preprocessor as p
from util import red, cyan


def set_up_stop_words() -> set:
    """Download and return English stop words."""
    try:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))
    except Exception as e:
        red(f"[stopwords] Failed to load NLTK stop words: {e}")
        return set()


def remove_stop_words(tokens: list[str], _stop_words: set) -> list[str]:
    """Remove stop words from token list."""
    if not _stop_words:
        return tokens  # fallback: don't filter
    return [t for t in tokens if t.lower() not in _stop_words]


def preprocess_twitter_text(_text: str) -> str:
    """Clean tweet-specific artifacts from the text."""
    return p.clean(_text)


def sentiment_tokenizer(_text: str) -> list[str]:
    """Tokenize input using a custom regex pattern."""
    pattern = r"""(?x)(?:[A-Z]\.)+|\w+(?:[-']\w+)*|\$?\d+(?:\.\d+)?%?|\.\.\.|[][.,;"'?():-_`]"""
    return regexp_tokenize(_text, pattern)


def stem_text(tokens: list[str]) -> list[str]:
    """Stem tokens using PorterStemmer."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def process_text(_text: str, _stop_words: set = None, tweet: bool = True, verbose: bool = False) -> list[str] | None:
    """Full pipeline for preprocessing a single text."""
    try:
        _stop_words = _stop_words or set()
        if tweet:
            _text = preprocess_twitter_text(_text)
        tokens = sentiment_tokenizer(_text)
        filtered = remove_stop_words(tokens, _stop_words)
        stemmed = stem_text(filtered)
        if verbose:
            cyan(f"Text: {_text}")
            print("→ Tokenized:", tokens)
            print("→ Filtered:", filtered)
            print("→ Stemmed:", stemmed)
        return stemmed
    except Exception as e:
        red(f"Error processing text: {e}")
        return None


# =======================
# 🧪 BASIC TEST
# =======================
if __name__ == "__main__":
    stop_words = set_up_stop_words()
    text = "This is a sample tweet! #example @user123 😂"
    result = process_text(text, stop_words, tweet=True, verbose=True)
    print("Processed Text:", result)
