import onnxruntime as ort
import numpy as np
import time
import os

model_path = 'outputs/pacd_net_edge.onnx'
if not os.path.exists(model_path):
    print(f'FATAL: {model_path} not found.')
    exit(1)

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

dummy_visual = np.random.randn(1, 3, 224, 224).astype(np.float32)
dummy_physio = np.random.randn(1, 4).astype(np.float32)

inputs = {
    'visual': dummy_visual,
    'physio': dummy_physio,
}

print('===================================================')
print('           PHASE C: EDGE LATENCY BENCHMARK         ')
print('===================================================')
print(f'Model      : {model_path} ({os.path.getsize(model_path)/1024/1024:.2f} MB)')
print(f'Simulation : 1 Thread (Strict Edge NPU Constraints)')
print('---------------------------------------------------')

for _ in range(100):
    session.run(None, inputs)

iters = 1000
latencies = []
for _ in range(iters):
    t0 = time.perf_counter()
    session.run(None, inputs)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)

latencies_np = np.array(latencies)
avg_latency_ms = latencies_np.mean()
p50 = np.percentile(latencies_np, 50)
p95 = np.percentile(latencies_np, 95)
p99 = np.percentile(latencies_np, 99)
min_lat = latencies_np.min()
max_lat = latencies_np.max()
fps = 1000.0 / avg_latency_ms

print('===================================================')
print('                  FINAL TELEMETRY                  ')
print('===================================================')
print(f'Iterations    : {iters}')
print(f'Avg Latency   : {avg_latency_ms:.2f} ms / inference')
print(f'Min Latency   : {min_lat:.2f} ms')
print(f'P50 Latency   : {p50:.2f} ms')
print(f'P95 Latency   : {p95:.2f} ms')
print(f'P99 Latency   : {p99:.2f} ms')
print(f'Max Latency   : {max_lat:.2f} ms')
print(f'Estimated FPS : {fps:.2f} frames/sec')
print('===================================================')
