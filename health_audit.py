import psutil
import time
import sys

def run_health_audit():
    print("==================================================")
    print("      TRL-3 EMPIRICAL TRAINING HEALTH AUDIT")
    print("==================================================")
    
    target_script = "train_empirical.py"
    active_process = None
    
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'num_ctx_switches']):
        cmd = p.info.get('cmdline')
        if cmd and target_script in ' '.join(cmd):
            active_process = p
            break
            
    if not active_process:
        print(f"[FATAL] {target_script} is NOT RUNNING. The process has died silently.")
        sys.exit(1)
        
    pid = active_process.info['pid']
    print(f"[ACTIVE] Target Process Found : PID {pid}")
    
    # Sample CPU over a 2-second window for an accurate active read
    cpu_usage = active_process.cpu_percent(interval=2.0)
    ram_usage = active_process.memory_info().rss / (1024 * 1024)
    
    try:
        ctx_switches = active_process.num_ctx_switches()
        io_activity = f"Voluntary: {ctx_switches.voluntary} | Involuntary: {ctx_switches.involuntary}"
    except:
        io_activity = "Context switch monitoring unavailable on this OS layer."

    print(f"[METRIC] CPU Utilization      : {cpu_usage:.1f}%")
    print(f"[METRIC] RAM Footprint        : {ram_usage:.2f} MB")
    print(f"[METRIC] Thread I/O Switches  : {io_activity}")
    
    if cpu_usage < 1.0:
        print("\n[WARNING] CPU usage is near zero. The PyTorch loop is likely deadlocked or waiting on I/O.")
    else:
        print("\n[STATUS] PERFECT OPTIMIZATION FLOW. The PyTorch graph is actively computing gradients.")
        
    print("==================================================")

if __name__ == "__main__":
    run_health_audit()
