# ========================= Presented by: Reza Saadatyar (2026) ================================
# ============================= E-mail: R.Saadatyar90@gmail.com ==================================

import platform

import psutil
import torch


def verify_hardware():
    """
    Print basic system information and select the PyTorch compute device.

    Returns:
        device (torch.device): CUDA when available, otherwise CPU.
    """
    print("⏳ Initializing hardware verification protocol...\n")

    # -------------------------------- 1. System Information --------------------------------
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    os_name = f"{platform.system()} {platform.release()}"
    cpu_name = platform.processor() or platform.machine()

    print(f"🖥️ OS: {os_name}")
    print(f"🧠 CPU: {cpu_name}")
    print(f"💾 RAM: {ram_gb:.1f} GB")

    # -------------------------------- 2. PyTorch Device ------------------------------------
    print(f"\n🔧 PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        print("✅ PyTorch CUDA is available.")
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 VRAM: {gpu_vram:.2f} GB")
        print(f"🔹 CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA is not available. CPU mode selected.")

    print(f"\n✅ Selected device: {device}")
    return device
