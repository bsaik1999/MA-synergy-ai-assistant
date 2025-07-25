# M&A Synergy Prediction Engine

🔍 A full-stack LLM + Bayesian-powered pipeline for analyzing merger and acquisition (M&A) deals.

## 🚀 What It Does

- Embeds company business descriptions using **Sentence-BERT**
- Calculates **semantic synergy scores** between parent and acquired companies
- Trains a **Bayesian logistic regression model** to predict acquisition success
- Simulates **fair value using Monte Carlo**, accounting for risk
- Provides:
  - ✅ A **FastAPI endpoint** for programmatic access
  - 📊 A **Streamlit dashboard** for uploading files and viewing interactive insights

## 🧠 Core Tech Stack

- Sentence-BERT (semantic embeddings)
- PyMC for Bayesian inference
- Monte Carlo simulations
- FastAPI (REST API)
- Streamlit (UI + visual analytics)

## 📁 File Structure

```bash
├── MAAI.py                # Main ML pipeline: embedding, modeling, simulation
├── app/
│   ├── api.py             # FastAPI app
│   ├── dashboard.py       # Streamlit UI
├── data/
│   └── acquisitions_update_2021.csv
├── requirements.txt
├── README.md
