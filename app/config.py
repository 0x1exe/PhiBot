import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTIMODAL_MODEL_PATH = "models/multimodal_phi_small_gpu.pth"