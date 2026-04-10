# Contributing to Patient Readmission Prediction

Thank you for your interest in contributing! Follow the steps below to get started.

## 1. Fork and Clone the Repository

```bash
git clone https://github.com/<your-username>/Patient-Readmission-Prediction-.git
cd Patient-Readmission-Prediction-
```

## 2. Set Up the Environment

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run the Pipeline

Generate the dataset and run the full analysis:

```bash
python data/generate_data.py
python notebooks/full_analysis.py
```

## 4. Make Your Changes

- Work on a descriptive feature branch:
  ```bash
  git checkout -b feature/your-feature-name
  ```
- Keep changes focused and well-documented.
- Follow the existing code style.

## 5. Submit a Pull Request

```bash
git add .
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
```

Then open a Pull Request against the `main` branch on GitHub and describe what your change does and why.

## Code of Conduct

Please be respectful and constructive in all interactions.
