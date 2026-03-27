import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBEVGlobalAlign(nn.Module):
    """
    GraphBEV-style global alignment block:
      1) predict dense 2D offsets from concatenated image/lidar BEV features
      2) deform lidar BEV features via differentiable grid sampling
      3) optimize an auxiliary MSE alignment loss against fused BEV target
    """
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg

        self.img_channel = int(self.model_cfg.get('IMG_CHANNEL', 80))
        self.lidar_channel = int(self.model_cfg.get('LIDAR_CHANNEL', 128))
        in_channel = int(self.model_cfg.get('IN_CHANNEL', self.img_channel + self.lidar_channel))
        out_channel = int(self.model_cfg.get('OUT_CHANNEL', self.lidar_channel))
        self.max_offset_pix = float(self.model_cfg.get('MAX_OFFSET_PIX', 4.0))
        self.loss_weight = float(self.model_cfg.get('LOSS_WEIGHT', 0.05))

        self.mm_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        self.offset_conv = nn.Conv2d(in_channel, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.deform_conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _get_base_grid(batch_size, height, width, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing='ij'
        )
        base_grid = torch.stack([x, y], dim=-1)
        return base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def forward_features(self, img_bev, lidar_bev, compute_loss=False):
        cat_bev = torch.cat([img_bev, lidar_bev], dim=1)
        offset = self.offset_conv(cat_bev)
        b, _, h, w = offset.shape
        offset = torch.tanh(offset) * self.max_offset_pix
        offset_x = offset[:, 0] / max((w - 1) / 2.0, 1.0)
        offset_y = offset[:, 1] / max((h - 1) / 2.0, 1.0)
        offset_grid = torch.stack([offset_x, offset_y], dim=-1)

        base_grid = self._get_base_grid(b, h, w, offset.device, offset.dtype)
        sample_grid = base_grid + offset_grid
        aligned_lidar = F.grid_sample(
            lidar_bev, sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        aligned_lidar = self.deform_conv(aligned_lidar)

        loss = None
        if compute_loss:
            mm_bev = self.mm_conv(cat_bev)
            loss = F.mse_loss(aligned_lidar, mm_bev) * self.loss_weight
        return aligned_lidar, loss

