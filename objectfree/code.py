import torch
import numpy as np

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device('cpu') #if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Test small tensor to GPU
test_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
test_t = torch.from_numpy(test_img).float().to(device)
print("Tensor to GPU OK!")

# Test unfold (tương tự code entropy)
pad = 2
padded = torch.nn.functional.pad(test_t.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='constant')  # Add batch/channel dims for unfold
patches = padded.unfold(2, 5, 1).unfold(3, 5, 1)  # Simulate window_size=5
print("Unfold OK! Shape:", patches.shape)