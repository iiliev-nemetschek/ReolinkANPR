FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 5001
VOLUME ["/config", "/app/data", "/app/logs", "/root/.cache"]

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "run.py"]
