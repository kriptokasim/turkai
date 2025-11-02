FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","1","--timeout-keep-alive","5","--log-level","info"]
