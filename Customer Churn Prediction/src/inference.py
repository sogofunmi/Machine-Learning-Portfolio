import sklearn
import joblib


def load_model(model_path):
    dv, model = joblib.load(model_path)

    return dv, model

def predict_test(df, dv, model):
    dicts = df.to_dict(orient="records")
    X = dv.transform(dicts)

    y_pred = model.predict(X)
    probas = model.predict_proba(X)[:,1]
    
    return y_pred, probas