import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_universality():
    # Data from your experiments
    data = [
        {"Model": "Qwen-72B", "Type": "Standard Judge", "Liar Win Rate": 50.0},
        {"Model": "Mistral-12B", "Type": "Standard Judge", "Liar Win Rate": 86.0},
        {"Model": "Llama-8B", "Type": "Standard Judge", "Liar Win Rate": 60.0},
        {"Model": "Qwen-72B", "Type": "Safety Judge", "Liar Win Rate": 0.0}
    ]
    df = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))

    # Color code: Red for vulnerable, Green for safe
    colors = ["#e74c3c" if x > 10 else "#2ecc71" for x in df["Liar Win Rate"]]

    ax = sns.barplot(data=df, x="Model", y="Liar Win Rate", palette=colors, edgecolor="black")
    
    plt.title("Universality: Sycophancy Breaks Multiple Models", fontsize=20, fontweight='bold', pad=20)
    plt.ylabel("Liar Win Rate (%)")
    plt.ylim(0, 100)

    for i, p in enumerate(ax.patches):
        ax.text(p.get_x() + p.get_width()/2., p.get_height() + 2, 
                f"{int(p.get_height())}%", ha="center", fontweight='bold')

    plt.tight_layout()
    plt.savefig("universality_chart.png", dpi=300)
    print("Saved universality_chart.png")

if __name__ == "__main__":
    plot_universality()