import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAlign(nn.Module):
    def __init__(self,model_cfg) -> None:
        super(GlobalAlign, self).__init__()
        self.model_cfg = model_cfg
        self.img_channel = self.model_cfg.get('IMG_CHANNEL', 80)
        self.lidar_channel = self.model_cfg.get('LIDAR_CHANNEL', 80)
        in_channel = self.model_cfg.get('IN_CHANNEL', self.img_channel + self.lidar_channel)
        out_channel = self.model_cfg.get('OUT_CHANNEL', self.lidar_channel)
        self.max_offset_pix = float(self.model_cfg.get('MAX_OFFSET_PIX', 4.0))
        self.loss_weight = float(self.model_cfg.get('LOSS_WEIGHT', 0.05))
        self.conv = nn.Sequential(
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
            lidar_bev, sample_grid, mode='bilinear',
            padding_mode='zeros', align_corners=True
        )
        deformed_feature = self.deform_conv(aligned_lidar)
        loss = None
        if compute_loss:
            mm_bev = self.conv(cat_bev)
            loss = self.calculate_loss(deformed_feature, mm_bev) * self.loss_weight
        return deformed_feature, loss

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']
        deformed_feature, loss = self.forward_features(
            img_bev, lidar_bev, compute_loss=self.training
        )
        batch_dict['spatial_features'] = deformed_feature
        if loss is not None:
            batch_dict['loss_global_align'] = loss
        return batch_dict
    

    def calculate_loss(self, deformed_feature, mm_bev):
        loss_fn = nn.MSELoss()
        loss = loss_fn(deformed_feature, mm_bev)
        return loss

