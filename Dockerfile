FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN unzip -o models.zip -d models && rm models.zip

# Указываем порт для Hugging Face (7860)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]