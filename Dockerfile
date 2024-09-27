FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY download.py .
RUN python download.py

COPY ./ ./
EXPOSE 8001

CMD ["python", "main.py"]
