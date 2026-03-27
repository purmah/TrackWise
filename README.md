# TrackWise — Personal Finance AI Manager

> An end-to-end AI-powered expense intelligence system that automatically classifies transactions, forecasts future spending, and generates personalized budgeting recommendations.

---

## Overview

Managing personal finances is tedious — scattered transactions, no smart categorization, and zero visibility into future spending. **TrackWise** solves this by combining NLP, time-series forecasting, and LLM-driven optimization into a single unified pipeline.

---

## Features

### 🧾 Expense Categorization
- Converts raw transaction descriptions into dense semantic vectors using **SentenceBERT (MiniLM)**
- Classifies transactions using an **SVM + XGBoost ensemble** across 20 spending categories
- Achieves **~90% accuracy** on short and noisy real-world transaction text
- Outperforms traditional TF-IDF and keyword-based approaches

### 📈 Spending Forecasting
- Builds **60-day sliding windows** with rolling features (7/14/30-day stats + calendar features)
- Uses a **Random Forest + Gradient Boosting + Ridge ensemble** to predict average daily spend over the next 30 days
- Applies **beta distribution modeling + hypothesis testing** for robust normalization of unseen sequences
- Supports city-based cost-of-living scaling and multi-currency output

### 🧠 Budget Optimization
- Compresses transaction history into interpretable financial signals (net spend, volatility, category concentration, forecast trend)
- Feeds structured financial context into a **Gemini LLM** to generate personalized budgeting recommendations
- Distinguishes between structural overspending, instability-driven risk, and category concentration risk

### 📧 Email Parsing
- Parses financial transaction emails using IMAP
- Extracts transaction amounts and categorizes purchases automatically

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | SentenceBERT (MiniLM all-MiniLM-L6-v2) |
| Categorization | SVM + XGBoost Ensemble |
| Forecasting | Random Forest, Gradient Boosting, Ridge |
| Optimization | Gemini LLM |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## Model Performance

| Model | Accuracy |
|-------|----------|
| SVM | 90.0% |
| XGBoost | 86.96% |
| Ensemble (SVM + XGBoost) | 89.13% |

---

## Project Structure

```
TrackWise/
├── app.py                  # Streamlit entrypoint
├── demo.py                 # Launch script
├── requirements.txt
├── Categorization/         # NLP categorization module
│   ├── categorizer.py
│   └── Categorization.ipynb
├── Forecasting/            # Time-series forecasting module
│   ├── Model.py
│   ├── conversion.py
│   └── city_index.py
├── Optimization/           # LLM budget optimization
│   └── Optimize.py
├── email_category/         # Email transaction categorizer
├── email_parser/           # Email parser
├── Data_Preprocessing/     # EDA and preprocessing notebooks
└── data/                   # Datasets
```

---

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/purmah/TrackWise.git
cd TrackWise
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create `secrets.yaml`
Create a `secrets.yaml` file at the root of the project:
```yaml
Gemini_API_key: YOUR_GEMINI_KEY
Currency_API_key: YOUR_CURRENCY_KEY
RAPIDAPI_KEY: YOUR_RAPIDAPI_KEY
Gmail_Paskey: YOUR_GMAIL_APP_PASSWORD
```

### 5. Run the app
```bash
python demo.py
```

Open your browser at `http://localhost:8501`

---

## Usage

- **Categorization tab** — Enter any transaction description and get an instant predicted category with confidence scores
- **Forecasting tab** — Generate synthetic data or upload your own CSV to get a 30-day expense forecast scaled to your city and currency
- **Optimization tab** — Ask free-form questions about your spending and get AI-generated budgeting advice

---

## Future Work

- Integration with Plaid API for real-time bank transaction ingestion
- Frontend dashboard using Next.js + Supabase
- AWS Lambda deployment for scalable model inference
- Expanded dataset for improved rare-category accuracy