FROM python:3.10.16-bookworm

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY models/xgboost.pkl ./
COPY src/api.py ./

EXPOSE 8091

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:8091", "api:app"]