# Run the whole pipeline on any cloud VM / container service:
#   docker build -t acc-lab-ca .  &&  docker run --rm -v $PWD/results:/app/results acc-lab-ca
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
