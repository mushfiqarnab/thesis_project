import os
import importlib

def run_edge_audit():
    print("==================================================")
    print("      EDGE COMPRESSION & HARDWARE PROFILER")
    print("==================================================")
    
    required_packages = ['torch_pruning', 'onnx', 'onnxruntime', 'psutil']
    all_passed = True
    
    for p in required_packages:
        try:
            m = importlib.import_module(p)
            version = getattr(m, "__version__", "installed")
            print(f"[OK] {p.ljust(15)} : v{version}")
        except ImportError:
            print(f"[FATAL] {p.ljust(15)} : FAILED TO LOAD")
            all_passed = False
            
    if not all_passed:
        print("\n[!] ENVIRONMENT AUDIT FAILED. Resolve dependencies before Phase 4.")
        return
        
    import onnxruntime as ort
    import torch
    
    print("\n--- ONNX Hardware Simulation Constraints ---")
    providers = ort.get_available_providers()
    print(f"Execution Providers Available: {providers}")
    if 'CPUExecutionProvider' not in providers:
        print("[WARNING] CPUExecutionProvider missing. Edge simulation may be inaccurate.")
        
    # Check PyTorch default threads (must be constrained during benchmark)
    default_threads = torch.get_num_threads()
    print(f"PyTorch Default CPU Threads  : {default_threads}")
    if default_threads > 1:
        print("[INFO] PyTorch multi-threading detected. The final onnx_benchmark.py MUST force intra_op_num_threads=1 to simulate a valid IoT Edge device.")
        
    print("==================================================")
    print("SYSTEM READY FOR HARDWARE-AWARE STRUCTURED PRUNING")
    print("==================================================")

if __name__ == "__main__":
    run_edge_audit()
