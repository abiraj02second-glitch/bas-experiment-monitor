FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=3000

WORKDIR /app

# Runtime libraries required by OpenCV, MediaPipe and offline speech.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        espeak-ng \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-hosted.txt ./
RUN pip install --upgrade pip && pip install -r requirements-hosted.txt

COPY . .

RUN mkdir -p recordings uploads

EXPOSE 3000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-3000}"]
