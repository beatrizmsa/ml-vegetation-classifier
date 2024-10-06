# Recebe um dataset e faz alteracoes e avalia cada  dataset  com diversos modelos printa os resultados
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def grafic_visualization(data, title, columns, type):
    if type == 'numerical':
        columns = columns.drop('Id')
    if type == 'categorical':
        columns = columns.drop('Vegetation_Type')
    fig, axes = plt.subplots((len(columns) + 2) // 3, 3, figsize=(15, 20))

    axes = axes.flatten()
    for i, col in enumerate(columns):

        if type == 'numerical':
            data.boxplot(column=col, by='Vegetation_Type', ax=axes[i])
        if type == 'categorical':
            pd.crosstab(data[col], data['Vegetation_Type']).plot(kind='bar', ax=axes[i], stacked=True)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
