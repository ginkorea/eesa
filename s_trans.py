from sentence_transformers import SentenceTransformer
import numpy as np


def cosine_similarity(vector1, vector2):
    dot = np.dot(vector1, vector2)
    norm_1 = np.linalg.norm(vector1)
    norm_2 = np.linalg.norm(vector2)
    cos_sim = dot / (norm_1 * norm_2)
    return cos_sim


class SentenceComparison:

    def __init__(self, s1, s2, l_model='paraphrase-MiniLM-L6-v2'):
        self.s1 = s1
        self.s2 = s2
        self.sentences = [s1, s2]
        self.model = SentenceTransformer(l_model)
        self.embeddings = self.model.encode(self.sentences)
        self.cos_sim = cosine_similarity(self.embeddings[0], self.embeddings[1])

    def __repr__(self):
        return self.cos_sim


sent_1 = "hello bob, what are you up to?"
sent_2 = "why did we decide to do this?"

comp = SentenceComparison(sent_1, sent_2)
print(comp)