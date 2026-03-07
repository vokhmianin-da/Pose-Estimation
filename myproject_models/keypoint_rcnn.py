import torch
import torchvision.transforms as T
from src.pose_utils import load_model, get_keypoints_from_pil

class KeypointRCNNModel:
    """Обёртка для Keypoint R-CNN."""
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = load_model(self.device)

    def get_keypoints(self, img_pil, conf_threshold=0.9):
        """Возвращает ключевые точки для изображения."""
        return get_keypoints_from_pil(img_pil, self.model, self.device, conf_threshold)