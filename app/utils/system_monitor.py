import psutil
try:
    import GPUtil
    has_gpu = True
except ImportError:
    has_gpu = False
import os
import time
from datetime import datetime

def get_system_resources():
    """Get system resource usage (CPU, RAM, GPU, Disk)"""
    # Get CPU info
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    # Get memory info
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_total = memory.total
    memory_used = memory.used
    
    # Get disk info
    disk = psutil.disk_usage('/')
    disk_percent = disk.percent
    disk_total = disk.total
    disk_used = disk.used
    
    # Get GPU info if available
    gpu_info = []
    if has_gpu:
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,  # Convert to percentage
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                    'temperature': gpu.temperature
                })
        except Exception as e:
            print(f"Error getting GPU info: {str(e)}")
    
    # Get recordings storage info
    recordings_path = os.path.join('storage', 'recordings')
    if os.path.exists(recordings_path):
        recordings_disk = psutil.disk_usage(recordings_path)
        recordings_percent = recordings_disk.percent
        recordings_total = recordings_disk.total
        recordings_used = recordings_disk.used
    else:
        recordings_percent = 0
        recordings_total = 0
        recordings_used = 0
    
    # Return all system information
    return {
        'timestamp': datetime.now().isoformat(),
        'cpu': {
            'percent': cpu_percent,
            'count': cpu_count,
            'freq': cpu_freq.current if cpu_freq else 0
        },
        'memory': {
            'percent': memory_percent,
            'total': memory_total,
            'used': memory_used
        },
        'disk': {
            'percent': disk_percent,
            'total': disk_total,
            'used': disk_used
        },
        'recordings': {
            'percent': recordings_percent,
            'total': recordings_total,
            'used': recordings_used
        },
        'gpu': gpu_info
    }

def get_system_stats():
    """Alias for get_system_resources for backward compatibility"""
    return get_system_resources()

def log_system_resources(log_file='logs/resources.log', interval=60):
    """Log system resources to a file at specified interval"""
    while True:
        resources = get_system_resources()
        
        # Format log line
        log_line = (
            f"{resources['timestamp']}, "
            f"CPU: {resources['cpu']['percent']}%, "
            f"RAM: {resources['memory']['percent']}%, "
            f"Disk: {resources['disk']['percent']}%"
        )
        
        # Add GPU info if available
        if resources['gpu']:
            for i, gpu in enumerate(resources['gpu']):
                log_line += f", GPU{i}: {gpu['load']:.1f}% ({gpu['memory_percent']:.1f}%)"
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Write to log file
        with open(log_file, 'a') as f:
            f.write(log_line + "\n")
        
        # Wait for next interval
        time.sleep(interval)