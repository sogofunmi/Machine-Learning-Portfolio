import joblib
from flask import Flask, request, jsonify

model_path = "xgboost.pkl"
dv, model = joblib.load(model_path)


app = Flask("Customer Churn")

@app.route("/", methods=["GET"])
def home():
    return "API is running!"

@app.route('/', methods=['POST'])

def predict():
    customer = request.get_json()

    X = dv.transform([customer])

    proba = round(model.predict_proba(X)[0,1], 2)
    churn = proba >= 0.5

    result = {
        "Probability": float(proba),
        "Churn": bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8091)