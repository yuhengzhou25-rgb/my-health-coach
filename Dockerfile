FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY wechat_bot.py zhou_health_dossier.md ./

ENV PORT=8000

CMD ["sh", "-c", "uvicorn wechat_bot:app --host 0.0.0.0 --port ${PORT}"]