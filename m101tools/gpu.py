import pynvml as nv

def init(): 
    """Initialize GPU monitoring

    Requires: None.
    Returns: handle.
    """
    try: 
        nv.nvmlInit() 
        # gpu_count = nv.nvmlDeviceGetCount() 
        # print(f"[GPU] Found {gpu_count} GPU(s)") 

        # Get handle 
        handle = nv.nvmlDeviceGetHandleByIndex(0)
        raw_name = nv.nvmlDeviceGetName(handle)
        gpu_name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
        print(f"[GPU] Using {gpu_name}")

        return handle 
    except Exception as e: 
        print(f"[WARNING] GPU monitoring is not available: {e}")
        return None 

def memory(handle):
    """Get GPU memory usage in GB. 

    Requires: handle
    Returns: used_gb (float)
    """

    if handle is None: 
        return 0.0 

    try: 
        mem_info = nv.nvmlDeviceGetMemoryInfo(handle) 
        used_gb = mem_info.used / 1024**3   # Bytes to GB
        return round(used_gb, 2)
    except Exception as e: 
        print(f"[WARNING] Failed to get GPU memory: {e}")
        return 0.0 

def utilization(handle): 
    """Get GPU utilization

    Requires: handle 
    Returns: util.gpu (float) 
    """

    if handle is None: 
        return 0.0 

    try: 
        util = nv.nvmlDeviceGetUtilizationRates(handle) 
        return util.gpu
    except Exception as e: 
        return 0.0 
