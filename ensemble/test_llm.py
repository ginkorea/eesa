# test_llm.py

from ensemble.llm_classifier import label_row
from util import green

if __name__ == "__main__":

    # Example usage of the label_row function
    test_texts = ["This is one of the best products I've ever used. Highly recommend it!",
                    "The service was terrible and the staff were rude. I will never come back.",
                    "It's an average product, nothing special but it works.", "I absolutely love this! It exceeded my expectations.",
                    "I was disappointed with the quality. It didn't meet my standards.",
                    "The product is decent, but I've seen better options out there.", "I can't believe how bad this was. I expected so much more.",
                    "The best product I have purchased from Trump. If only it could make him just go away.", "Trump"]

    for text in test_texts:
        print(f"\n🔍 Processing: {text}")
        result = label_row(text, num_classifiers=3, verbose=True)
        print(f"Sentiment Score     : {result[0]:.3f}")
        print(f"Confidence          : {result[1]:.3f}")
        print(f"Explanation Quality : {result[2]:.3f}")
        print(f"Explanation         : {result[3]}")
