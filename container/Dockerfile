FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py pdf_loader.py qa.py ./

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
