FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src \
    ATTENDANCE_DB=/data/attendance.db ATTENDANCE_MODEL_DIR=/app/models

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY src ./src
# Models are fetched at build time and checked against the SHA-256 pinned in config/settings.yaml.
RUN python -m attendance models

RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app
VOLUME ["/data"]
EXPOSE 8000
CMD ["python", "-m", "attendance", "serve", "--port", "8000"]
