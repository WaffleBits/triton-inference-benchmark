import time
import numpy as np
import matplotlib.pyplot as plt
import ray
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ray for distributed computing
ray.init()

class TritonBenchmark:
    def __init__(self, server_url="localhost:8000", model_name="resnet50_trt_fp16", 
                 input_shape=(1, 3, 224, 224), num_requests=200, concurrent_requests=10):
        self.server_url = server_url
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_requests = num_requests
        self.concurrent_requests = concurrent_requests
        self.client = httpclient.InferenceServerClient(url=server_url)
        
    @ray.remote
    def _send_inference_request(self, input_data, retries=3):
        """Send a single inference request with retry mechanism"""
        inputs = [httpclient.InferInput("input", input_data.shape, 
                                      np_to_triton_dtype(input_data.dtype))]
        inputs[0].set_data_from_numpy(input_data)

        for attempt in range(retries):
            try:
                start_time = time.time()
                response = self.client.infer(self.model_name, inputs)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                return {"status": "success", "latency": latency}
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}: Error {e}")
                if attempt == retries - 1:
                    return {"status": "failed", "error": str(e)}
                time.sleep(1)  # Wait before retry
    
    def generate_dummy_input(self):
        """Generate dummy input data for benchmarking"""
        return np.random.rand(*self.input_shape).astype(np.float32)
    
    def run_benchmark(self):
        """Execute distributed benchmark"""
        logger.info(f"Starting benchmark with {self.num_requests} requests...")
        
        futures = []
        results = {"latencies": [], "errors": 0, "successes": 0}
        
        # Launch distributed inference requests
        for _ in range(self.num_requests // self.concurrent_requests):
            batch = [self.generate_dummy_input() for _ in range(self.concurrent_requests)]
            futures += [self._send_inference_request.remote(self, data) for data in batch]
        
        # Collect results
        for result in ray.get(futures):
            if result["status"] == "success":
                results["latencies"].append(result["latency"])
                results["successes"] += 1
            else:
                results["errors"] += 1
        
        # Calculate metrics
        if results["latencies"]:
            avg_latency = np.mean(results["latencies"])
            p95_latency = np.percentile(results["latencies"], 95)
            p99_latency = np.percentile(results["latencies"], 99)
            throughput = 1000 * self.concurrent_requests / avg_latency
            
            metrics = {
                "average_latency_ms": float(avg_latency),
                "p95_latency_ms": float(p95_latency),
                "p99_latency_ms": float(p99_latency),
                "throughput_fps": float(throughput),
                "successful_requests": results["successes"],
                "failed_requests": results["errors"]
            }
            
            self._save_results(metrics)
            self._plot_results(results["latencies"])
            
            return metrics
        return None

    def _save_results(self, metrics):
        """Save benchmark results to JSON"""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = output_dir / f"benchmark_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Results saved to {output_file}")

    def _plot_results(self, latencies):
        """Generate visualization of latency distribution"""
        plt.figure(figsize=(10, 5))
        plt.hist(latencies, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel("Latency (ms)")
        plt.ylabel("Frequency")
        plt.title(f"Distributed Inference Latency Distribution\n{self.model_name}")
        plt.grid(True)
        
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(output_dir / f"latency_distribution_{timestamp}.png")
        plt.close()

if __name__ == "__main__":
    benchmark = TritonBenchmark()
    metrics = benchmark.run_benchmark()
    
    if metrics:
        logger.info("\nBenchmark Results:")
        logger.info(f"Average Latency: {metrics['average_latency_ms']:.2f} ms")
        logger.info(f"P95 Latency: {metrics['p95_latency_ms']:.2f} ms")
        logger.info(f"P99 Latency: {metrics['p99_latency_ms']:.2f} ms")
        logger.info(f"Throughput: {metrics['throughput_fps']:.2f} inferences/sec")
        logger.info(f"Success Rate: {metrics['successful_requests']}/{metrics['successful_requests'] + metrics['failed_requests']}")
    
    ray.shutdown()