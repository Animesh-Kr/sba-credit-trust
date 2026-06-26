FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
ENV SBA_BUNDLE=/app/models/serving_bundle.joblib
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
