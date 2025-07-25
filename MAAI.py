# Updated MAAI.py with Monte Carlo simulation for synergy valuation

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pymc as pm
import arviz as az
from tqdm import tqdm

tqdm.pandas()

def simulate_synergy_fair_value(synergy_score, base_price, risk_factor, delta=50_000_000, lam=20_000_000, iterations=10000, synergy_cap=100_000_000, synergy_volatility=0.2):
    synergy_mean = synergy_score * synergy_cap
    synergy_std = synergy_mean * synergy_volatility
    simulated_synergy = np.random.normal(loc=synergy_mean, scale=synergy_std, size=iterations)
    simulated_fair_values = base_price + delta * (simulated_synergy / synergy_cap) - lam * risk_factor
    summary = {
        "mean_fair_value": np.mean(simulated_fair_values),
        "std_fair_value": np.std(simulated_fair_values),
        "VaR_5": np.percentile(simulated_fair_values, 5),
        "VaR_95": np.percentile(simulated_fair_values, 95),
        "p_fair_value_gt_price": np.mean(simulated_fair_values > base_price)
    }
    return summary

def run_mna_pipeline(file_path, save_trace=True, train_model=True):
    print(f"🚀 Running M&A pipeline on {file_path} ...")
    df = pd.read_csv(file_path)
    df = df[df['Business'].astype(str).str.strip() != '-'].copy()
    text_columns = ['Parent Company', 'Acquired Company', 'Business', 'Category', 'Derived Products', 'Country']
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    if 'Acquisition Price' in df.columns:
        df['Acquisition Price'] = df['Acquisition Price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['Acquisition Price'] = pd.to_numeric(df['Acquisition Price'], errors='coerce').fillna(0)
    else:
        df['Acquisition Price'] = 0
    df.dropna(subset=['Business', 'Acquired Company'], inplace=True)

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Encoding business descriptions...")
    df['Business Embedding'] = df['Business'].progress_apply(lambda x: model.encode(x))

    print("Computing parent embeddings...")
    parent_embeddings = df.groupby('Parent Company')['Business Embedding'].apply(lambda x: np.mean(np.stack(x), axis=0)).to_dict()

    def compute_synergy(row):
        parent_vec = parent_embeddings.get(row['Parent Company'])
        if parent_vec is None:
            return np.nan
        return cosine_similarity([parent_vec], [row['Business Embedding']])[0][0]

    print("Calculating synergy scores...")
    df['Synergy Score'] = df.progress_apply(compute_synergy, axis=1)

    parent_main_cat = df.groupby('Parent Company')['Category'].agg(lambda x: x.mode()[0] if not x.mode().empty else '-')
    df['Parent Main Category'] = df['Parent Company'].map(parent_main_cat)
    df['Category Similarity'] = (df['Category'] == df['Parent Main Category']).astype(int)
    df['Risk Factor'] = ((df['Country'] == '-') | (df['Acquisition Price'] == 0) | (df['Business'].str.len() < 10)).astype(int)

    model_df = df[df['Synergy Score'].notna()].copy()
    if 'Acquisition Price' in model_df.columns:
        threshold = model_df['Acquisition Price'].median()
        model_df['Successful'] = (model_df['Acquisition Price'] > threshold).astype(int)
    else:
        model_df['Successful'] = 0

    X = model_df[['Synergy Score', 'Category Similarity', 'Risk Factor']]
    y = model_df['Successful'].values

    trace = None
    if train_model:
        try:
            trace = az.from_netcdf("pymc_trace.nc")
            print("✅ Loaded existing Bayesian trace.")
        except FileNotFoundError:
            trace = None
        if trace is None:
            print("Training Bayesian logistic regression...")
            with pm.Model() as logistic_model:
                beta_0 = pm.Normal("Intercept", mu=0, sigma=10)
                beta = pm.Normal("beta", mu=0, sigma=5, shape=X.shape[1])
                mu = beta_0 + pm.math.dot(X.values, beta)
                p = pm.Deterministic("p", pm.math.sigmoid(mu))
                y_obs = pm.Bernoulli("y_obs", p=p, observed=y)
                trace = pm.sample(1000, tune=1000, target_accept=0.95, progressbar=True)
            if save_trace:
                az.to_netcdf(trace, "pymc_trace.nc")
                print("✅ Bayesian trace saved to pymc_trace.nc")

    sector_base_price = model_df.groupby('Category')['Acquisition Price'].median().to_dict()
    model_df['Base Price'] = model_df['Category'].map(sector_base_price).fillna(model_df['Acquisition Price'].median())
    simulation_results = model_df.apply(lambda row: simulate_synergy_fair_value(
        synergy_score=row['Synergy Score'],
        base_price=row['Base Price'],
        risk_factor=row['Risk Factor']), axis=1)
    model_df['Simulated Fair Value Mean'] = simulation_results.apply(lambda x: x['mean_fair_value'])
    model_df['Simulated Fair Value VaR_5'] = simulation_results.apply(lambda x: x['VaR_5'])
    model_df['Simulated Fair Value VaR_95'] = simulation_results.apply(lambda x: x['VaR_95'])
    model_df['P(Fair Value > Price)'] = simulation_results.apply(lambda x: x['p_fair_value_gt_price'])

    df.to_pickle("processed_acquisition_data.pkl")
    model_df.to_pickle("model_acquisition_data.pkl")
    print("✅ Data saved with Monte Carlo simulation: processed_acquisition_data.pkl & model_acquisition_data.pkl")

    return df, model_df, trace
