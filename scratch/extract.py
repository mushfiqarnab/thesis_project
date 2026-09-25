import json
import glob
from pathlib import Path

reports_dir = Path('C:/Users/USERAS/thesis_project/outputs/reports')

print('--- Master Benchmark ---')
master = reports_dir / 'equitas_rcmf_master_benchmark_report.json'
if master.exists():
    data = json.loads(master.read_text())
    print('Autonomous evaluations:')
    for k, v in data['autonomous_evaluations'].items():
        print(f"  {k}: acc {v['acc']:.4f}, dp {v['dp_abs']:.4f}, eo {v['eo_max_gap']:.4f}, cf {v['cf_gap']:.4f}")
    
    print('Demographic evaluations:')
    for k, v in data['demographic_evaluations'].items():
        print(f"  {k}: acc {v['acc']:.4f}, dp {v['dp_abs']:.4f}, eo {v['eo_max_gap']:.4f}, cf {v['cf_gap']:.4f}")

print('\n--- Edge Benchmark ---')
edge = reports_dir / 'equitas_rcmf_edge_benchmark_report.json'
if edge.exists():
    data = json.loads(edge.read_text())
    print(f"fp32 mean: {data['latency_fp32_ms']['mean']:.4f} ms")
    print(f"fp32 p95: {data['latency_fp32_ms']['p95']:.4f} ms")
    print(f"int8 mean: {data['latency_int8_ms']['mean']:.4f} ms")
    print(f"int8 p95: {data['latency_int8_ms']['p95']:.4f} ms")

