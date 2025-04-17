"""
Visualization utilities for printing or plotting matrix states.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def print_matrix(A, step=None):
    """
    Nicely print matrix A with optional step label.
    """
    if step is not None:
        print(f"--- Step {step} ---")
    # Display matrix without indices or headers
    df = pd.DataFrame(A)
    print(df.to_string(index=False, header=False))
    print()

def plot_matrix(A, step=None):
    """
    Plot matrix A (optional) for visualization. Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(np.array(A), cmap='viridis')
    fig.colorbar(cax)
    if step is not None:
        ax.set_title(f"Step {step}")
    return fig
