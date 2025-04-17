"""
Report generator for Gaussian elimination project.

Generates Markdown report with intermediate matrices.
"""

import numpy as np

def generate_report(intermediates, solution, filepath="Project3_report.md"):
    """
    Generate a Markdown report of the elimination steps and solution.
    :param intermediates: list of numpy.ndarray for each elimination step
    :param solution: numpy.ndarray solution vector
    :param filepath: output file path
    """
    with open(filepath, 'w') as f:
        f.write("# Gaussian Elimination Report\n\n")
        f.write("## Elimination Steps\n\n")
        for i, A in enumerate(intermediates):
            f.write(f"### Step {i}\n\n")
            f.write("```\n")
            f.write(np.array2string(A, precision=6, separator=', '))
            f.write("\n```\n\n")
        f.write("## Solution\n\n")
        f.write("```\n")
        f.write(np.array2string(solution, precision=6, separator=', '))
        f.write("\n```\n")
