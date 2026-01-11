import json
import random
import pandas as pd
from datasets import load_dataset

def prep_data():
    print("Loading TruthfulQA...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    
    # We will use 50 topics to ensure we finish in time
    subset = dataset.select(range(50)) 
    
    tasks = []
    for item in subset:
        # Structure the debate setup
        task = {
            "id": item["type"] + "_" + str(random.randint(1000, 9999)),
            "topic": item["question"],
            "truth_premise": item["best_answer"],
            # Pick the first common misconception as the assigned Lie
            "lie_premise": item["incorrect_answers"][0] 
        }
        tasks.append(task)
    
    output_path = "data/debate_configs.json"
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Created {len(tasks)} debate configs at {output_path}")
    print(f"Example Topic: {tasks[0]['topic']}")
    print(f"Assigned Lie: {tasks[0]['lie_premise']}")

if __name__ == "__main__":
    prep_data()