FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY .env.example .env

RUN mkdir -p logs

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
