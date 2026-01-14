import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load dataset (robust to current working directory)
data_path = Path(__file__).parent / 'User_knowledge.csv'
u_df = pd.read_csv(data_path)

# Optional: encode label if you need it later (not used for EM input)
u_df['UNS'].replace(['very_low', 'Low', 'Middle', 'High'], [0, 1, 2, 3], inplace=True)

# Use features only for EM
X = u_df.drop(columns=['UNS']).values

# Scale features for stable GMM fitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit EM (Gaussian Mixture)
gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',
    init_params='random',
    n_init=10,
    random_state=42
)
labels = gmm.fit_predict(X_scaled)

# MLE parameters (mixture weights, means, covariances in scaled space)
print("Weights:", gmm.weights_)
print("Means (scaled):\n", gmm.means_)
print("Covariances (scaled):\n", gmm.covariances_)

# AIC and BIC on the same data used to fit
print("AIC:", gmm.aic(X_scaled))
print("BIC:", gmm.bic(X_scaled))

