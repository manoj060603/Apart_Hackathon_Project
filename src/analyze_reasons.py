import json
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download stopwords if not present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def analyze_reasons():
    with open("data/judgments.json", "r") as f:
        data = json.load(f)
    
    # Filter for cases where the Sycophant Liar won
    sycophant_wins = [d for d in data if d['condition'] == 'Sycophant' and d['winner_role'] == 'Lie']
    
    print(f"Analyzing {len(sycophant_wins)} Sycophant Victories...")
    
    text = " ".join([d['reason'].lower() for d in sycophant_wins])
    words = [w for w in text.split() if w.isalpha() and w not in stop_words]
    
    common = Counter(words).most_common(20)
    print("\nTop words in Judge's reasoning when Sycophant wins:")
    for word, count in common:
        print(f"{word}: {count}")

    # Let's print 3 random examples of the Judge's reasoning
    print("\n--- EXAMPLE JUDGE REASONING ---")
    for item in sycophant_wins[:3]:
        print(f"Topic: {item['topic']}")
        print(f"Reason: {item['reason']}\n")

if __name__ == "__main__":
    analyze_reasons()