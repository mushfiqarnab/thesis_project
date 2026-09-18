"""
onnx_benchmark.py

A strictly honest, executable benchmark harness using ONNX Runtime.

This script measures the actual hardware-accelerated latency of an ONNX model,
bypassing PyTorch's heavy Python overhead. It forces 1-thread execution by 
default to simulate constrained edge IoT microcontrollers, and runs warmup 
passes to ensure the CPU cache and ONNX graph optimizations are settled 
before timing.

Since we don't have a pre-exported `.onnx` file of the pruned network in 
this sandbox, this script uses `onnx.helper` to dynamically generate a 
dummy linear graph (if a model path isn't provided), proving the benchmark 
machinery works and producing real, verifiable stdout.
"""

import os
import time
import argparse
import numpy as np
import onnxruntime as ort

def create_dummy_onnx(path="dummy_edge_model.onnx"):
    """Generates a simple 3-layer MLP in ONNX format without PyTorch."""
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print("ONNX library not found. Cannot generate dummy model.")
        return False

    # Define a small 3-layer MLP simulating a compressed edge network
    # X (1, 128) -> Dense -> (1, 64) -> Dense -> (1, 32) -> Dense -> (1, 2)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 128])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])

    # Weights
    W1_np = np.random.randn(128, 64).astype(np.float32)
    W2_np = np.random.randn(64, 32).astype(np.float32)
    W3_np = np.random.randn(32, 2).astype(np.float32)

    W1 = helper.make_tensor('W1', TensorProto.FLOAT, [128, 64], W1_np.tobytes(), raw=True)
    W2 = helper.make_tensor('W2', TensorProto.FLOAT, [64, 32], W2_np.tobytes(), raw=True)
    W3 = helper.make_tensor('W3', TensorProto.FLOAT, [32, 2], W3_np.tobytes(), raw=True)

    # Nodes
    node1 = helper.make_node('MatMul', inputs=['X', 'W1'], outputs=['H1'], name='MatMul1')
    node2 = helper.make_node('Relu', inputs=['H1'], outputs=['A1'], name='Relu1')
    node3 = helper.make_node('MatMul', inputs=['A1', 'W2'], outputs=['H2'], name='MatMul2')
    node4 = helper.make_node('Relu', inputs=['H2'], outputs=['A2'], name='Relu2')
    node5 = helper.make_node('MatMul', inputs=['A2', 'W3'], outputs=['Y'], name='MatMul3')

    graph = helper.make_graph(
        nodes=[node1, node2, node3, node4, node5],
        name='dummy_mlp',
        inputs=[X],
        outputs=[Y],
        initializer=[W1, W2, W3]
    )

    model = helper.make_model(graph, producer_name='onnx_benchmark_harness')
    onnx.save(model, path)
    return True

def benchmark_onnx(model_path, num_threads=1, warmup=50, iters=1000):
    print("=" * 70)
    print(f"ONNX RUNTIME BENCHMARK HARNESS")
    print("=" * 70)
    
    # Configure ONNX Runtime for Edge Simulation
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1 # Keep inter-op to 1 for strict sequential edge simulation
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Use CPU execution provider (closest approximation to standard edge IoT without NPU)
    try:
        sess = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return

    # Introspect model signature and generate dummy inputs matching the exact shape/type
    inputs = {}
    for inp in sess.get_inputs():
        shape = inp.shape
        # Resolve dynamic batch dimensions to 1
        shape = [1 if (s is None or isinstance(s, str)) else s for s in shape]
        
        if inp.type == 'tensor(float)':
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        elif inp.type in ['tensor(int64)', 'tensor(int32)']:
            inputs[inp.name] = np.random.randint(0, 10, size=shape).astype(np.int64)
        else:
            inputs[inp.name] = np.zeros(shape, dtype=np.float32) # Fallback

    print(f"Model loaded successfully: {model_path}")
    print(f"Input tensors mapped:      { {k: v.shape for k, v in inputs.items()} }")
    print(f"Graph optimizations:       ORT_ENABLE_ALL")
    print(f"Intra-op threads:          {num_threads}")

    print(f"\nExecuting {warmup} warmup passes...")
    for _ in range(warmup):
        sess.run(None, inputs)

    print(f"Executing {iters} benchmark iterations...")
    times = []
    
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    
    mean_ms = times.mean()
    p50_ms = np.percentile(times, 50)
    p95_ms = np.percentile(times, 95)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0

    print("\n" + "-" * 70)
    print("REAL EXECUTED HARDWARE TELEMETRY")
    print("-" * 70)
    print(f"Mean Latency : {mean_ms:7.3f} ms")
    print(f"P50 Latency  : {p50_ms:7.3f} ms")
    print(f"P95 Latency  : {p95_ms:7.3f} ms")
    print(f"Throughput   : {fps:7.1f} FPS")
    print("-" * 70)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dummy_edge_model.onnx", help="Path to ONNX model")
    parser.add_argument("--threads", type=int, default=1, help="Number of CPU threads")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=1000, help="Benchmark iterations")
    args = parser.parse_args()

    # If the model doesn't exist, try to generate a dummy one to prove the harness works
    if not os.path.exists(args.model):
        print(f"Model {args.model} not found. Attempting to generate a synthetic dummy graph...")
        success = create_dummy_onnx(args.model)
        if not success:
            return

    benchmark_onnx(args.model, num_threads=args.threads, warmup=args.warmup, iters=args.iters)

if __name__ == "__main__":
    main()
