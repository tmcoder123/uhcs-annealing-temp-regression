import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

def train_optimized_hybrid(train_df, val_df, test_df, train_feats, val_feats, test_feats, n_pca=16):
    # PCA Latent compression
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feats)
    val_scaled = scaler.transform(val_feats)
    test_scaled = scaler.transform(test_feats)

    pca = PCA(n_components=n_pca, random_state=42)
    train_pca = pca.fit_transform(train_scaled)
    val_pca = pca.transform(val_scaled)
    test_pca = pca.transform(test_scaled)

    # Encode Tabular Physics
    cool_dummies = pd.get_dummies(train_df['cool_method'], prefix='cool', dtype=float)
    train_cols = cool_dummies.columns

    def build_matrix(sub_df, pca_feats):
        t = np.log1p(sub_df['anneal_time'].values).reshape(-1, 1)
        m = np.log1p(sub_df['magnification'].values).reshape(-1, 1)
        c = pd.get_dummies(sub_df['cool_method'], prefix='cool', dtype=float)
        c_aligned = c.reindex(columns=train_cols, fill_value=0.0).values
        return np.hstack([pca_feats, t, m, c_aligned]), sub_df['target'].values

    X_train, y_train = build_matrix(train_df, train_pca)
    X_val, y_val = build_matrix(val_df, val_pca)
    X_test, y_test = build_matrix(test_df, test_pca)

    weights = compute_sample_weight('balanced', train_df['microconstituent'])

    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=4,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        early_stopping_rounds=40,
        eval_metric="mae"
    )

    model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_test)
    return {
        'model': model,
        'r2': r2_score(y_test, preds),
        'mae': mean_absolute_error(y_test, preds),
        'y_test': y_test,
        'y_pred': preds
    }
