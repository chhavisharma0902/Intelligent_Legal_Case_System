import re
from collections import defaultdict
from collections import Counter

def extract_keywords(text, top_n=5):
    words = text.split()
    freq = Counter(words)
    keywords = [w for w, _ in freq.most_common(top_n)]
    return ", ".join(keywords)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)

    stopwords = set([
        "the","is","in","and","to","of","for","on","with",
        "court","judge","case","law","section"
    ])

    words = text.split()
    words = [w for w in words if w not in stopwords and len(w) > 2]

    return " ".join(words)


def build_cooccurrence(text, window_size=2):
    words = text.split()
    cooc = defaultdict(int)

    for i in range(len(words)):
        for j in range(i+1, min(i+window_size+1, len(words))):
            pair = (words[i], words[j])
            cooc[pair] += 1

    return cooc


def cooccurrence_score(text1, text2):
    cooc1 = build_cooccurrence(text1)
    cooc2 = build_cooccurrence(text2)

    score = 0
    for pair in cooc1:
        if pair in cooc2:
            score += min(cooc1[pair], cooc2[pair])

    return score