# Recebe um dataset e faz alteracoes e avalia cada  dataset  com diversos modelos printa os resultados
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


colors = ["#FFB3BA",   "#FFDFBA",   "#FFFFBA",   "#BAFFC9", "#BAE1FF",  "#FFB3E6",  "#B3FFDA",  "#CAB2FF", "#FFB3FF",
          "#FFC1E3",  "#CCE5FF",  "#B2F0E6",  "#FFD1B2",  "#FFFF99",  "#D1C4E9",   "#FFE0B2",  "#F8BBD0",  "#DCEDC8"]


def boxplot_visualization(data, columns, title):
    n_cols = min(5, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(data=data, y=col, ax=axes[i], color=colors[i])

    for j in range(len(columns), len(axes)):
        axes[j].axis('off')

    plt.tight_layout([0, 0, 1, 0.95])
    plt.suptitle(title, fontsize=16)
    plt.show()


def barplot_visualization(data, columns, title):
    n_cols = min(5, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        label = data.groupby(col).size()
        sns.barplot(x=label.index, y=label.values, ax=axes[i], color=colors[i])
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(len(columns), len(axes)):
        axes[j].axis('off')

    plt.tight_layout([0, 0, 1, 0.95])
    plt.suptitle(title, fontsize=16)
    plt.show()
