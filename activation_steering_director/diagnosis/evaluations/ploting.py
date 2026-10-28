import matplotlib.pyplot as plt
import numpy as np

# Categories and metrics
categories = ["P", "I", "C", "O"]
metrics = ["precision", "recall", "f1"]

# Unsteered and Steered data dictionaries
unsteered_data = {
    "P": {"precision": 0.506128, "recall": 0.864538, "f1": 0.528682},
    "I": {"precision": 0.413755, "recall": 0.606513, "f1": 0.426432},
    "C": {"precision": 0.394649, "recall": 0.830232, "f1": 0.396053},
    "O": {"precision": 0.577384, "recall": 0.820502, "f1": 0.547142}
}

steered_data = {
    "P": {"precision": 0.511042, "recall": 0.859223, "f1": 0.530111},
    "I": {"precision": 0.414588, "recall": 0.609895, "f1": 0.428492},
    "C": {"precision": 0.360619, "recall": 0.832945, "f1": 0.364038},
    "O": {"precision": 0.558210, "recall": 0.824189, "f1": 0.531767}
}

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight'
})

for metric in metrics:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    
    unsteered_vals = [unsteered_data[cat][metric] for cat in categories]
    steered_vals = [steered_data[cat][metric] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, unsteered_vals, width, label='Unsteered', color='#4C72B0', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, steered_vals, width, label='Steered', color='#DD8452', edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'PICO Evaluation: {metric.capitalize()}')
    ax.set_xticks(x)
    ax.set_xticklabels(['Population', 'Intervention', 'Comparator', 'Outcome'])
    
    # Legend in the top-right corner
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    # Value labels with 3 decimal places
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
            
    # Increased headroom multiplier to 1.35 so legend never overlaps bars
    max_val = max(max(unsteered_vals), max(steered_vals))
    ax.set_ylim(0, max_val * 1.35)
    
    plt.tight_layout()
    plt.savefig(f'comparison_{metric}_steer_unsteer.pdf')
    plt.close()