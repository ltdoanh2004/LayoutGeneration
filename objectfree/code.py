""" import torch
import numpy as np
import gc
import time
# Khởi tạo CUDA trước khi làm bất cứ điều gì
if torch.cuda.is_available():
    # Force CUDA initialization
    torch.zeros(1).cuda()
    # Sleep để đảm bảo initialization hoàn tất
    time.sleep(1)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# Force garbage collection
gc.collect()
torch.cuda.empty_cache()

try:
    
    # Test small tensor to GPU
    test_img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    test_t = torch.from_numpy(test_img).float().to(device)
    print("Tensor to GPU OK!")

    # Test unfold (tương tự code entropy)
    pad = 2
    padded = torch.nn.functional.pad(test_t.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='constant')  # Add batch/channel dims for unfold
    print("Padding OK! Shape:", padded.shape)

    patches = padded.unfold(2, 5, 1).unfold(3, 5, 1)  # Simulate window_size=5
    print("Unfold OK! Shape:", patches.shape)

except Exception as e:
    print("Error occurred:", e) """


import os
import sys

print("=== PyTorch CUDA Diagnosis ===")

print("Step 1: Import NumPy")
import numpy as np
print("✅ NumPy imported")

print("\nStep 2: Import PyTorch")
import torch
print("✅ PyTorch imported")
print(f"PyTorch version: {torch.__version__}")

# Save original environment variable
original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

# Check if CUDA is available with current settings
print("\nStep 3: Check current CUDA availability")
cuda_available_current = torch.cuda.is_available()
print(f"CUDA available (current settings): {cuda_available_current}")

# Temporarily remove any CUDA restrictions to check true availability
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]
    print("Removed CUDA_VISIBLE_DEVICES restriction for testing")
    
    # Need to reimport torch.cuda to reset availability
    import importlib
    importlib.reload(torch.cuda)

# Check true CUDA availability
print("\nStep 4: Check true CUDA availability")
cuda_available_true = torch.cuda.is_available()
print(f"CUDA actually available: {cuda_available_true}")

if not cuda_available_true:
    print("❌ CUDA is not available on this system!")
    print("Possible causes:")
    print("  - Missing CUDA drivers")
    print("  - PyTorch built without CUDA support")
    print("  - Incompatible CUDA version")
    print("\nCheck with:")
    print("  nvidia-smi")
    print("  python -c \"import torch; print(torch.version.cuda)\"")
    sys.exit(1)

# Test basic CPU operations first
print("\nStep 5: Test CPU tensor operations")
cpu_tensor = torch.tensor([1, 2, 3])
print(f"✅ CPU tensor: {cpu_tensor}")

# Try carefully initializing CUDA
print("\nStep 6: Carefully initialize CUDA")
try:
    # Try to get device count
    device_count = torch.cuda.device_count()
    print(f"✅ GPU count: {device_count}")
    
    # Try to get properties
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"✅ GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")
    
    # Try to initialize CUDA context
    print("\nStep 7: Initialize CUDA context")
    x_cpu = torch.tensor([1.0, 2.0, 3.0])
    x_gpu = x_cpu.cuda()
    print(f"✅ GPU tensor: {x_gpu}")
    
    # Try to synchronize
    torch.cuda.synchronize()
    print("✅ CUDA synchronize successful")
    
    # Test numpy → torch → GPU conversion
    print("\nStep 8: Test numpy → torch → GPU")
    np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    torch_tensor = torch.from_numpy(np_array)
    gpu_tensor = torch_tensor.cuda()
    print(f"✅ GPU tensor from numpy: {gpu_tensor}")
    
except Exception as e:
    print(f"❌ CUDA test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ CUDA diagnosis complete - everything is working correctly!")

# Restore original environment if needed
if original_cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
    print(f"Restored CUDA_VISIBLE_DEVICES={original_cuda_visible_devices}")