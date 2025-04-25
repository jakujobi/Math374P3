# Gaussian Elimination Explorer – Project Summary

**Target Audience:** Recruiters & Future Employers

## Overview

This repository showcases a robust implementation of Gaussian elimination with scaled partial pivoting, designed for clarity, correctness, and ease of use. It includes:

- A core solver packaged in `src/gauss_sp.py` with input validation, error handling, and optional step-by-step logging.
- A Streamlit-based web app (`streamlit_app.py`) providing an interactive UI for entering matrices, visualizing elimination steps, and viewing solutions.
- Custom styling (`assets/custom.css`) to enhance the user experience.

## Key Achievements

- **Algorithmic Rigor:** Developed and tested a textbook Gaussian elimination algorithm with scaled pivoting to ensure numerical stability on small systems (2×2 up to 4×4).
- **Modular Design:** Organized code into a clear structure (`src/`, root scripts, and config) to separate concerns and support reuse.
- **Interactive Interface:** Used Streamlit to build an intuitive web front-end, enabling non-technical users and educators to explore the solver dynamically.
- **Quality Assurance:** Wrote comprehensive unit tests with `pytest` to validate solver correctness against NumPy and handle edge cases (singular matrices, dimension mismatches).
- **Packaging & Configuration:** Managed dependencies and metadata via `pyproject.toml` and `streamlit.toml`, ensuring reproducible environments and easy deployment.

## Tech Stack

- **Python 3.10+**: Core language.
- **NumPy**: Efficient numerical operations.
- **Streamlit**: Rapid UI development for data apps.
- **pytest**: Automated testing.
- **Poetry/pyproject.toml**: Dependency management.
- **CSS**: Custom styling for the Streamlit app.

## Project Structure

```
Math374P3/
├── assets/
│   └── custom.css       # Stylesheet for Streamlit UI
├── src/
│   └── gauss_sp.py      # Core solver implementation
├── streamlit_app.py     # Interactive web application
├── tests/
│   └── test_gauss_sp.py # Unit tests for solver
├── pyproject.toml       # Project metadata & dependencies
├── streamlit.toml       # Streamlit configuration
└── LICENSE              # MIT License
```

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run unit tests:
   ```bash
   pytest -q
   ```
3. Launch the web app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Future Enhancements

- Support larger systems (n×n for n > 4).
- Add CI/CD (GitHub Actions) for automated testing and deployment.
- Improve UI with custom charts for step visualization.
- Package solver as an installable library on PyPI.

---

 

## Resume Entry

 

**Gaussian Elimination Explorer**, course project — Apr 2025

- Implemented Gaussian elimination with scaled partial pivoting in Python, ensuring numerical stability for small systems
- Built a Streamlit web app for interactive visualization of elimination steps and solutions
- Authored comprehensive pytest-based unit tests verifying correctness against NumPy and handling edge cases
- Managed project configuration and dependencies with pyproject.toml, streamlit.toml, and custom CSS for UI enhancements

 
