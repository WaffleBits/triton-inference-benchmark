# Distributed Inference Benchmarking Tool

## 🚀 Overview
A sophisticated distributed inference benchmarking tool designed for NVIDIA Triton Inference Server. This project demonstrates expertise in distributed systems, deep learning optimization, and production-grade AI infrastructure.

## ✨ Key Features
- Distributed inference using Ray for parallel processing
- TensorRT FP16 optimization support
- Robust error handling and retry mechanisms
- Comprehensive performance metrics (latency, throughput, P95, P99)
- Real-time visualization of latency distribution
- Containerized deployment with Docker
- CI/CD pipeline with GitHub Actions

## 🛠 Technical Stack
- Python 3.10+
- NVIDIA Triton Inference Server
- Ray for distributed computing
- TensorRT for optimized inference
- Docker for containerization
- GitHub Actions for CI/CD

## 📊 Performance Metrics
- Average Latency (ms)
- P95/P99 Latency
- Throughput (inferences/second)
- Success/Error Rate
- Latency Distribution Visualization

## 🔧 Setup & Installation

### Prerequisites
```bash
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Runtime
- Python 3.10+
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/triton-inference-benchmark.git
cd triton-inference-benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Build Docker image:
```bash
docker build -t triton-benchmark .
```

## 🚀 Usage

### Running with Docker
```bash
docker run --gpus all --network host triton-benchmark
```

### Running locally
```bash
python benchmark.py
```

## 📈 Output
The tool generates:
- JSON files with detailed metrics
- Latency distribution plots
- Console logs with key performance indicators

## 🔍 Advanced Features
- Configurable number of concurrent requests
- Customizable retry mechanisms
- Support for different model architectures
- Real-time performance monitoring
- Distributed load generation

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
MIT License

## 🔗 Links
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [Ray Documentation](https://docs.ray.io/)
- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)