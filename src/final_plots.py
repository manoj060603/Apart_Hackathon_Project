import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_final_chart():
    # --- 1. THE DATA ---
    # Based on your final results
    data = [
        {"Phase": "Baseline", "Condition": "Control (Polite)", "Win Rate": 34.0, "Type": "Standard Judge"},
        {"Phase": "Attack",   "Condition": "Sycophant (Flattery)", "Win Rate": 50.0, "Type": "Standard Judge"},
        {"Phase": "Defense",  "Condition": "Constitutional Defense", "Win Rate": 0.0,  "Type": "Safety Judge"},
        {"Phase": "Evasion",  "Condition": "Stealth (Fake Studies)", "Win Rate": 6.0,  "Type": "Safety Judge"}
    ]
    
    df = pd.DataFrame(data)

    # --- 2. STYLE SETUP ---
    sns.set_theme(style="whitegrid", context="talk") # "talk" context makes fonts big for slides
    plt.figure(figsize=(12, 7))

    # Custom Colors
    # Grey for Control, Red for Danger, Green for Safety, Orange for Attempted Evasion
    palette = ["#95a5a6", "#c0392b", "#27ae60", "#f39c12"]

    # --- 3. CREATE PLOT ---
    ax = sns.barplot(
        data=df, 
        x="Condition", 
        y="Win Rate", 
        palette=palette,
        edgecolor="black",
        linewidth=2
    )

    # --- 4. ANNOTATIONS ---
    # Add percentage labels on top of bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        # Special handling for 0.0 to make it visible
        if height == 0:
            label = "0%"
            y_pos = 1
        else:
            label = f"{int(height)}%"
            y_pos = height + 1.5
            
        ax.text(
            p.get_x() + p.get_width() / 2., 
            y_pos, 
            label, 
            ha="center", 
            fontweight='bold', 
            fontsize=18,
            color='black'
        )

    # Add a "Danger Zone" background shading
    plt.axhspan(40, 60, facecolor='red', alpha=0.05)
    plt.text(3.3, 52, "Danger Zone\n(Coin Flip)", color='red', fontsize=12, ha='center')

    # --- 5. TITLES AND LABELS ---
    plt.title("The Arms Race: Sycophancy vs. Constitutional AI", fontsize=22, fontweight='bold', pad=20)
    plt.ylabel("Liar Win Rate (%)", fontsize=16, fontweight='bold')
    plt.xlabel("") # Remove x-label as categories are self-explanatory
    plt.ylim(0, 65) # Give some headroom

    # Add a text box for the "Verbosity" finding
    textstr = '\n'.join((
        r'$\bf{Key\ Insight:}$',
        'Sycophants won by flattery,',
        'NOT by word count.',
        '(Correlation: -0.075)'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # --- 6. SAVE ---
    plt.tight_layout()
    plt.savefig("final_results_chart.png", dpi=300)
    print("Graph saved as 'final_results_chart.png'")

if __name__ == "__main__":
    generate_final_chart()