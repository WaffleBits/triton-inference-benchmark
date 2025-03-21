FROM nvcr.io/nvidia/tritonserver:24.02-py3

# Install required packages
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copy benchmark code
WORKDIR /app
COPY benchmark.py /app/

# Create directory for results
RUN mkdir -p /app/benchmark_results

ENTRYPOINT ["python", "benchmark.py"]