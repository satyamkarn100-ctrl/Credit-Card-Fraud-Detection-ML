import pandas as pd
import numpy as np
import joblib

def predict_fraud(df):
    # load artifacts
    config = joblib.load('model/inference_config.pkl')
    model  = joblib.load('model/fraud_detector_final.pkl')
    scaler = joblib.load('model/scaler.pkl')

    data = df.copy()

    # fix time to hours if present
    if 'Time' in data.columns:
        data['Elapsed_Hour'] = (data['Time'] // 3600).astype(int)
        data = data.drop(columns=['Time'])

    # drop target if it somehow sneaks in
    if 'Class' in data.columns:
        data = data.drop(columns=['Class'])

    # scale amount and time
    data[['Amount', 'Elapsed_Hour']] = scaler.transform(data[['Amount', 'Elapsed_Hour']])

    # enforce model's column order
    data = data[config['feature_cols']]

    # get preds
    probs = model.predict_proba(data)[:, 1]
    preds = (probs >= config['threshold']).astype(int)

    return pd.DataFrame({
        'fraud_prob': probs.round(4),
        'is_fraud': preds,
        'label': ['FRAUD' if p == 1 else 'NORMAL' for p in preds]
    })


if __name__ == "__main__":
    # quick test run
    try:
        # grab 5 random rows from local csv
        df = pd.read_csv('creditcard.csv', nrows=50)
        sample = df.sample(5).drop(columns=['Class'], errors='ignore')
        print("testing on local csv data...")

    except FileNotFoundError:
        print("csv not found, generating dummy data...")
        # fallback if csv isn't there
        dummy = {'Time': [100, 4000, 80000], 'Amount': [10.5, 500.0, 2.5]}
        for i in range(1, 29):
            dummy[f'V{i}'] = np.random.randn(3)
        sample = pd.DataFrame(dummy)

    # run inference
    res = predict_fraud(sample)
    print(res)
