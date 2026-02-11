import os 
import sys

class SetupChecker: 
    """Flexible ML project setup verification.
    
    Usage:
        # Simple — run everything with defaults
        checker = SetupChecker()
        checker.run_all()

        # Custom — pick what you need
        checker = SetupChecker(
            env_path="config/.env",
            model_path=r"F:\\_ai\\models\\faster-whisper-large-v3",
            dependencies=["fastapi", "faster_whisper", "uvicorn"]
        )
        checker.run_all()

        # Run individual checks
        checker.check_pytorch()
        checker.check_gpu_memory()

        # Access results programmatically
        results = checker.run_all()
        if results["pytorch"].get("cuda_available"):
            print("Ready for GPU training!")
    """

    def __init__(self, env_path=".env", model_path=None, dependencies=None): 
        """Initialise the setup checker.

        Args: 
            env_path: Path to .env file. Defaults to '.env'. 
            model_path: Path to model directory. If None, reads from MODEL_PATH env var.
            dependencies: List of module names to check. 
        """
        self.env_path = env_path 
        self.model_path = model_path 
        self.dependencies = dependencies or [] 
        self.results = {} 

    def check_python(self): 
        self.results["python"] = sys.version
        print(f"Python version: {sys.version}") 

    def check_pytorch(self): 
        try: 
            import torch
            info = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
            self.results["pytorch"] = info
            print(f"PyTorch version: {info['version']}")
            print(f"CUDA available: {info['cuda_available']}")
            print(f"CUDA version: {info['cuda_version']}")
            if info["gpu_model"]:
                print(f"GPU model: {info['gpu_model']}")
        except Exception as e:
            self.results["pytorch"] = {"error": str(e)}
            print(f"PyTorch error: {e}")

    def check_gpu_memory(self, device_index=0):
        try:
            import pynvml as nv
            nv.nvmlInit()
            handle = nv.nvmlDeviceGetHandleByIndex(device_index)
            mem_info = nv.nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem_info.free / 1024**3
            total_gb = mem_info.total / 1024**3
            self.results["gpu_memory"] = {"free_gb": free_gb, "total_gb": total_gb}
            print(f"GPU memory: {free_gb:.2f} / {total_gb:.2f} GB free")
        except Exception as e:
            self.results["gpu_memory"] = {"error": str(e)}
            print(f"GPU memory error: {e}")

    def check_env(self):
        if os.path.exists(self.env_path):
            with open(self.env_path, "r") as f:
                lines = [l for l in f if l.strip() and not l.strip().startswith("#")]
            self.results["env"] = {"found": True, "count": len(lines)}
            print(f"Env variables found: {len(lines)} entries")
        else:
            self.results["env"] = {"found": False}
            print("Unable to reach environment variables")

    def check_dotenv(self):
        try:
            from dotenv import load_dotenv
            load_dotenv(self.env_path)
            self.results["dotenv"] = {"loaded": True}
            print("Dotenv loaded successfully")
        except Exception as e:
            self.results["dotenv"] = {"error": str(e)}
            print(f"Dotenv error: {e}")

    def check_model_path(self):
        path = self.model_path or os.getenv("MODEL_PATH", "")
        if path and os.path.exists(path):
            self.results["model"] = {"found": True, "path": path}
            print(f"Model found: {path}")
        else:
            self.results["model"] = {"found": False, "path": path}
            print(f"Model not found: {path}")

    def check_dependencies(self):
        import importlib
        missing = []
        for dep in self.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)

        self.results["dependencies"] = {"missing": missing}
        if missing:
            print(f"Dependencies: ✗ missing: {', '.join(missing)}")
        else:
            print("Dependencies: ✓ All installed")

    def run_all(self):
        """Run all checks."""
        print("=" * 42)
        print("Server setup verification")
        print("=" * 42)

        self.check_python()
        self.check_pytorch()
        self.check_gpu_memory()
        self.check_env()
        self.check_dotenv()
        self.check_model_path()
        if self.dependencies:
            self.check_dependencies()

        print("=" * 42)
        print("End of checks")
        print("=" * 42)
        return self.results
