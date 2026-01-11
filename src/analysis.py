import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze():
    # 1. Load Judgments
    try:
        with open("data/judgments.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Judgments file not found.")
        return

    df = pd.DataFrame(data)
    
    # Filter valid results
    df = df[df['winner_role'].isin(['Truth', 'Lie'])]
    
    print(f"Total Valid Debates: {len(df)}")
    
    # 2. Calculate Win Rates
    # We want to know: How often does the LIE win in each condition?
    summary = df.groupby('condition')['winner_role'].value_counts(normalize=True).unstack().fillna(0)
    
    if 'Lie' not in summary.columns:
        print("The Lie never won! Check your judge prompt.")
        return

    print("\n--- WIN RATES (Likelihood of Lie Winning) ---")
    print(summary['Lie'] * 100)

    # 3. Create Chart
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Bar Chart
    ax = sns.barplot(x=df['condition'], y=(df['winner_role'] == 'Lie'), errorbar=None, palette="magma")
    
    ax.set_ylabel("Liar Win Rate")
    ax.set_title("The 'Liar's Dividend': Does Rhetoric Help Falsehoods Win?")
    ax.set_ylim(0, 1.0) # 0 to 100%
    
    # Add values on top of bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.savefig("results_chart.png")
    print("\nChart saved as 'results_chart.png'")

if __name__ == "__main__":
    analyze()