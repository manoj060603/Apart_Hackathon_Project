import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def validate():
    # Load transcripts (for word counts)
    with open("data/debate_transcripts.json", "r") as f:
        transcripts = json.load(f)
    
    # Load judgments (for winners)
    with open("data/judgments.json", "r") as f:
        judgments = json.load(f)
    
    # Create a lookup map for judgments
    # We need to map ID -> Winner Role
    winner_map = {j['id']: j['winner_role'] for j in judgments}
    condition_map = {j['id']: j['condition'] for j in judgments}

    data = []
    
    for t in transcripts:
        tid = t['id']
        if tid not in winner_map:
            continue
            
        winner = winner_map[tid]
        condition = condition_map[tid]
        
        # Calculate lengths
        len_truth = len(t['opening_truth'].split()) + len(t['rebuttal_truth'].split())
        len_lie = len(t['opening_lie'].split()) + len(t['rebuttal_lie'].split())
        
        # Did the Liar win?
        liar_won = (winner == 'Lie')
        
        data.append({
            "id": tid,
            "condition": condition,
            "truth_len": len_truth,
            "lie_len": len_lie,
            "liar_won": liar_won,
            "len_diff": len_lie - len_truth # Positive means Liar was verbose
        })

    df = pd.DataFrame(data)
    
    print("--- VERBOSITY ANALYSIS ---")
    print(df.groupby('condition')[['truth_len', 'lie_len']].mean())
    
    print("\n--- DOES LENGTH PREDICT WINNING? ---")
    # Correlation between length difference and winning
    corr = df['len_diff'].corr(df['liar_won'])
    print(f"Correlation between Liar being longer and Liar winning: {corr:.3f}")
    
    # PLOT
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='len_diff', y='condition', hue='liar_won', palette={True: "red", False: "blue"}, alpha=0.6)
    plt.axvline(0, color='grey', linestyle='--')
    plt.xlabel("Length Difference (Words) [Positive = Liar Longer]")
    plt.title("Win Distribution vs. Verbosity")
    plt.tight_layout()
    plt.savefig("verbosity_check.png")
    print("Saved verbosity_check.png")

if __name__ == "__main__":
    validate()