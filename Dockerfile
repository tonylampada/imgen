FROM python:3.12-slim
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./
EXPOSE 8001

CMD ["python", "main.py"]
