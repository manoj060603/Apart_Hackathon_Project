import pandas as pd
from datasets import load_dataset

print("Downloading TruthfulQA...")
# 'generation' split contains the questions and correct answers
dataset = load_dataset("truthful_qa", "generation", split="validation")

# Convert to simple JSON for our loop
# We need: Question, Best Answer (Truth), Correct Alternatives
df = pd.DataFrame(dataset)

# Let's save just the first 100 for the hackathon
targets = df[['question', 'best_answer']].head(100)
targets.columns = ['topic', 'truth_premise']

# Creating a placeholder for the "Lie" (we will generate these or manual fill later)
# For now, we will just save the raw data
targets.to_json("data/topics_raw.json", orient="records", indent=2)
print("Saved 100 topics to data/topics_raw.json")