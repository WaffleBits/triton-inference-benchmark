FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY benchmark.py /app/

RUN useradd --create-home --uid 10001 benchmark \
    && mkdir -p /app/benchmark_results \
    && chown -R benchmark:benchmark /app

USER benchmark

ENTRYPOINT ["python", "benchmark.py"]
