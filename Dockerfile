FROM python:3.9-slim

WORKDIR /app

# Устанавливаем unzip (это было пропущено в прошлый раз)
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

# Копируем файлы
COPY requirements.txt .
COPY models.zip .

# Распаковываем модели
RUN mkdir -p models && unzip -o models.zip -d models && rm models.zip

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем сервер
COPY server.py .

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]