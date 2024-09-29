import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import lightning.pytorch as pl
from metrics import ariel_score


# https://www.kaggle.com/code/vyacheslavefimov/quantile-loss-quantile-regression
def q_loss(quantiles, y_pred, target):
    losses = []
    for i, q in enumerate(quantiles):        
        errors = target - y_pred[..., i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
    losses = 2 * torch.cat(losses, dim=2)  # B x 283 x 3

    return losses.mean()

quantiles = [0.1587, 0.5, 0.8413]


class ArielModel(pl.LightningModule):
    def __init__(self, backbone='mobilenet_v3_small'):
        super().__init__()
        
        self.filter = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(3, 1), stride=(2, 1), bias=False),  # 波段独立
            # nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(2, 1), bias=False),
            nn.LeakyReLU()
        )
        
        if backbone == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(dropout=0.0, norm_layer=nn.Identity)
            self.backbone.classifier[3] = nn.Linear(in_features=1024, out_features=283*3, bias=True)
        
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(norm_layer=nn.Identity)
            self.backbone.fc = nn.Linear(in_features=512, out_features=283*3, bias=True)

        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(norm_layer=nn.Identity)
            self.backbone.classifier[0] = nn.Dropout(p=0.0, inplace=True)
            self.backbone.classifier[1] = nn.Linear(in_features=1280, out_features=283*3, bias=True)

        elif backbone == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(norm_layer=nn.Identity)
            self.backbone.classifier[0] = nn.Dropout(p=0.0, inplace=True)
            self.backbone.classifier[1] = nn.Linear(in_features=1280, out_features=283*3, bias=True)

        else:
            pass
    
    
    def forward(self, x):
        x = self.filter(x)
        x = self.backbone(x)
        return x.view(-1, 283, 3)  # B x 283 x 3
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        #augmentations: disturb channel and reverse the time
        if self.current_epoch > 0:
            for j in range(x.shape[-1]):  # B x 1 x T x 283
                if np.random.random() > 0.8:
                    x[..., j] = x[..., j] * (1 + np.random.randn() * 0.01)
            for k in range(x.shape[0]):
                if np.random.random() > 0.8:
                    x[k] = torch.flip(x[k], (1,))

        y_pred = self(x)  # B x 283 x 3
        loss = q_loss(quantiles, y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=0, last_epoch=-1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss'
            }
        }
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).detach().cpu()  # B x 283 x 3
        y = y.cpu()
        return y, y_pred
    
    
    def validation_epoch_end(self, outputs):
        all_label = []
        all_pred = []
        all_loss = []
        for y, y_pred in outputs:
            all_label.append(y.numpy())
            all_pred.append(y_pred.numpy())
            loss = q_loss(quantiles, y_pred, y)
            all_loss.append(loss)
        
        all_label = np.vstack(all_label)  # N x 283
        all_pred = np.vstack(all_pred)  # N x 283 x 3
        val_loss = np.mean(all_loss)
        
        all_pred_mean = all_pred[:, :, 1]  # N x 283
        all_pred_sigma = (all_pred[:, :, 2] - all_pred[:, :, 0]) # / 2  # N x 283
        
        metric = ariel_score(
            all_label,
            np.concatenate([all_pred_mean.clip(0), all_pred_sigma.clip(0)], axis=1),
            all_label.mean(),
            all_label.std(),
            sigma_true=1e-5
        )

        # 获取当前步的学习率
        optimizer = self.optimizers()
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        self.log('lr', current_lr)

        self.log('val_loss', loss)
        self.log('metric', metric)
    
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred.detach().cpu().numpy()


if __name__ == '__main__':
    # ut
    model = ArielModel()
    inputs = torch.randn(4, 1, 375, 283, dtype=torch.float32)
    outputs = model(inputs)
    print(outputs.size())