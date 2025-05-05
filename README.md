# Gaussian Elimination with Scaled Partial Pivoting

An interactive Streamlit application demonstrating Gaussian elimination with scaled partial pivoting, showing every internal arithmetic step.

## Features

- **Fixed Walkthrough**: Step-by-step demonstration of Gaussian elimination for 3×3 and 4×4 systems
- **Interactive Playground**: Enter your own systems or generate random ones
- **Detailed Visualization**: Row highlighting shows pivot selection and elimination steps
- **Export Options**: Download detailed reports in Markdown and PDF formats

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/jakujobi/pivot_demo.git
   cd pivot_demo
   ```

2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

3. For PDF export functionality, ensure you have either WeasyPrint or FPDF2 installed:
   ```
   pip install weasyprint fpdf2
   ```

## Usage

Run the Streamlit app:
```
streamlit run streamlit_app.py
```

## Project Structure

- `src/gauss_sp.py`: Core solver implementation with detailed step logging
- `streamlit_app.py`: Streamlit application with UI and visualization
- `assets/custom.css`: Custom styling for matrix highlighting
- `tests/test_gauss_sp.py`: Unit tests for the solver

## Requirements

- Python 3.10+
- NumPy
- Pandas
- Streamlit
- WeasyPrint or FPDF2 (for PDF export)

## License

MIT License
