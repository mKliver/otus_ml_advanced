# Base
FROM python:3.9-slim

RUN apt-get update
RUN apt-get install -y --no-install-recommends

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5005

# Run
CMD ["flask", "run", "--host=0.0.0.0", "--port=5005"]