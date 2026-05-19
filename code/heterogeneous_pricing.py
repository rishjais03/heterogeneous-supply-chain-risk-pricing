# ============================================================
# HETEROGENEOUS PRICING OF SUPPLY CHAIN RISK
# COMPLETE EMPIRICAL ASSET PRICING ANALYSIS
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

# ------------------------------------------------------------
# 2. LOAD DATASETS
# ------------------------------------------------------------

# Main financial dataset
financial_data = pd.read_excel("final_dataset (3)(1).xlsx")

# Supply chain weights
weights_data = pd.read_excel("weights(1).xlsx")

# Stock return data
stock_returns = pd.read_excel("stock_returns (2)(1).xlsx")

# Macroeconomic shock data
shock_data = pd.read_excel("shock_data (1)(1).xlsx")

# Fama-French factor data
ff_factors = pd.read_excel("ff_factors(1).xlsx")

# ------------------------------------------------------------
# 3. DATA CLEANING & PREPARATION
# ------------------------------------------------------------

# Standardize column names
financial_data.columns = financial_data.columns.str.lower()
weights_data.columns = weights_data.columns.str.lower()
stock_returns.columns = stock_returns.columns.str.lower()
shock_data.columns = shock_data.columns.str.lower()
ff_factors.columns = ff_factors.columns.str.lower()

# Remove duplicates
financial_data = financial_data.drop_duplicates()

# Convert dates if available
datasets = [
    financial_data,
    stock_returns,
    shock_data,
    ff_factors
]

for dataset in datasets:
    if 'date' in dataset.columns:
        dataset['date'] = pd.to_datetime(dataset['date'])

# ------------------------------------------------------------
# 4. CONSTRUCT MACROECONOMIC SHOCK VARIABLE
# ------------------------------------------------------------

# Standardize shock variable
if 'shock_value' in shock_data.columns:

    shock_data['standardized_shock'] = (
        shock_data['shock_value'] -
        shock_data['shock_value'].mean()
    ) / shock_data['shock_value'].std()

else:
    
    # Fallback if column names differ
    numeric_cols = shock_data.select_dtypes(include=np.number).columns
    
    shock_data['standardized_shock'] = (
        shock_data[numeric_cols[0]] -
        shock_data[numeric_cols[0]].mean()
    ) / shock_data[numeric_cols[0]].std()

print("\nShock Variable Constructed Successfully")

# ------------------------------------------------------------
# 5. MERGE NETWORK WEIGHTS WITH SHOCK DATA
# ------------------------------------------------------------

merged_network = pd.merge(
    weights_data,
    shock_data,
    how='left'
)

print("\nMerged Network Data")
print(merged_network.head())

# ------------------------------------------------------------
# 6. CONSTRUCT NETWORK SUPPLY CHAIN RISK (NSCR)
# ------------------------------------------------------------

# Automatically identify weight column
weight_cols = [
    col for col in merged_network.columns
    if 'weight' in col
]

if len(weight_cols) > 0:
    weight_col = weight_cols[0]
else:
    weight_col = merged_network.select_dtypes(include=np.number).columns[0]

# Weighted exposure calculation
merged_network['weighted_exposure'] = (
    merged_network[weight_col] *
    merged_network['standardized_shock']
)

# Automatically identify firm identifier
if 'firm_id' in merged_network.columns:
    firm_col = 'firm_id'
else:
    firm_col = merged_network.columns[0]

# Aggregate firm-level exposure
nscr = (
    merged_network
    .groupby(firm_col)['weighted_exposure']
    .sum()
    .reset_index()
)

nscr.rename(
    columns={
        'weighted_exposure': 'nscr',
        firm_col: 'firm_id'
    },
    inplace=True
)

print("\nNSCR Measure")
print(nscr.head())

# ------------------------------------------------------------
# 7. PREPARE FINANCIAL DATA
# ------------------------------------------------------------

# Standardize firm identifier
if 'firm_id' not in financial_data.columns:
    financial_data.rename(
        columns={financial_data.columns[0]: 'firm_id'},
        inplace=True
    )

# ------------------------------------------------------------
# 8. MERGE DATASETS
# ------------------------------------------------------------

# Merge NSCR
panel_data = pd.merge(
    financial_data,
    nscr,
    on='firm_id',
    how='left'
)

# Merge stock returns
if 'firm_id' not in stock_returns.columns:
    stock_returns.rename(
        columns={stock_returns.columns[0]: 'firm_id'},
        inplace=True
    )

panel_data = pd.merge(
    panel_data,
    stock_returns,
    on=['firm_id', 'date'],
    how='left'
)

# Merge Fama-French factors
panel_data = pd.merge(
    panel_data,
    ff_factors,
    on='date',
    how='left'
)

print("\nCombined Panel Dataset")
print(panel_data.head())

# ------------------------------------------------------------
# 9. HANDLE MISSING VALUES
# ------------------------------------------------------------

panel_data = panel_data.dropna()

# ------------------------------------------------------------
# 10. PANEL DATA STRUCTURE
# ------------------------------------------------------------

panel_data = panel_data.set_index(['firm_id', 'date'])

# ------------------------------------------------------------
# 11. DEFINE REGRESSION VARIABLES
# ------------------------------------------------------------

# Automatically identify stock return column
possible_return_cols = [
    'stock_return',
    'return',
    'returns',
    'ret'
]

return_col = None

for col in possible_return_cols:
    if col in panel_data.columns:
        return_col = col
        break

if return_col is None:
    return_col = panel_data.select_dtypes(include=np.number).columns[0]

# Dependent variable
y = panel_data[return_col]

# Base explanatory variables
regression_vars = ['nscr']

# Fama-French factors
ff_controls = [
    'mkt_rf',
    'smb',
    'hml',
    'rmw',
    'cma',
    'rf'
]

for factor in ff_controls:
    if factor in panel_data.columns:
        regression_vars.append(factor)

# Firm-level controls
firm_controls = [
    'size',
    'bm_ratio',
    'momentum',
    'volatility',
    'leverage'
]

for control in firm_controls:
    if control in panel_data.columns:
        regression_vars.append(control)

# Independent variable matrix
X = panel_data[regression_vars]

# Add constant
X = sm.add_constant(X)

# ------------------------------------------------------------
# 12. OLS REGRESSION
# ------------------------------------------------------------

ols_model = sm.OLS(y, X).fit()

print("\n================================================")
print("OLS REGRESSION RESULTS")
print("================================================")

print(ols_model.summary())

# ------------------------------------------------------------
# 13. PANEL FIXED EFFECTS REGRESSION
# ------------------------------------------------------------

fe_model = PanelOLS(
    y,
    X,
    entity_effects=True
)

fe_results = fe_model.fit(cov_type='clustered')

print("\n================================================")
print("FIXED EFFECTS REGRESSION RESULTS")
print("================================================")

print(fe_results.summary)

# ------------------------------------------------------------
# 14. ROBUSTNESS CHECKS
# ------------------------------------------------------------

robust_X = X.copy()

# Add squared momentum term if available
if 'momentum' in panel_data.columns:

    robust_X['momentum_squared'] = (
        panel_data['momentum'] ** 2
    )

robust_model = sm.OLS(y, robust_X).fit()

print("\n================================================")
print("ROBUSTNESS CHECK RESULTS")
print("================================================")

print(robust_model.summary())

# ------------------------------------------------------------
# 15. SUMMARY STATISTICS
# ------------------------------------------------------------

print("\n================================================")
print("SUMMARY STATISTICS")
print("================================================")

print(panel_data.describe())

# ------------------------------------------------------------
# 16. VISUALIZATION — NSCR DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

panel_data['nscr'].hist(bins=30)

plt.title("Distribution of Network Supply Chain Risk (NSCR)")
plt.xlabel("NSCR")
plt.ylabel("Frequency")

plt.show()

# ------------------------------------------------------------
# 17. VISUALIZATION — NSCR VS STOCK RETURNS
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    panel_data['nscr'],
    panel_data[return_col],
    alpha=0.5
)

plt.title("NSCR vs Stock Returns")
plt.xlabel("Network Supply Chain Risk (NSCR)")
plt.ylabel("Stock Returns")

plt.show()

# ------------------------------------------------------------
# 18. SAVE REGRESSION RESULTS
# ------------------------------------------------------------

coefficients = pd.DataFrame({
    'Variable': ols_model.params.index,
    'Coefficient': ols_model.params.values,
    'P-Value': ols_model.pvalues.values
})

coefficients.to_csv(
    "regression_results.csv",
    index=False
)

print("\nRegression results saved successfully.")

# ------------------------------------------------------------
# 19. FINAL CONCLUSION
# ------------------------------------------------------------

print("""
============================================================
ANALYSIS COMPLETED SUCCESSFULLY
============================================================

This notebook performed:

1. Financial data preparation
2. Macroeconomic shock construction
3. Network Supply Chain Risk (NSCR) estimation
4. Supply chain exposure aggregation
5. Panel-data construction
6. Fama-French factor integration
7. OLS regression estimation
8. Fixed-effects panel regression
9. Robustness testing
10. Empirical visualizations

The empirical analysis evaluates whether
supply chain risk is heterogeneously priced
across firms embedded within production networks.

Research Areas:
- Empirical Asset Pricing
- Production Networks
- Financial Econometrics
- Risk Modeling
- Quantitative Finance
============================================================
""")
