# NutriScan AI: container for Hugging Face Spaces (Docker SDK)
FROM python:3.11-slim

# OpenCV (pulled in by ultralytics) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics
WORKDIR $HOME/app

# CPU-only PyTorch (the default GPU build is several GB larger and unused here)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt gunicorn

# Download the EasyOCR Thai + English models at build time, not on every wake-up
RUN python -c "import easyocr; easyocr.Reader(['th', 'en'], gpu=False)"

COPY --chown=user . .

EXPOSE 7860
# One worker keeps a single copy of the AI models in memory; threads serve page/API requests
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300"]
