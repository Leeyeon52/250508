from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

def unet_training():
    kfold = KFold(
        n_splits=CONFIG['num_folds'], 
        random_state=CONFIG['random_seed'], 
        shuffle=True
    )

    for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(ps_meta_array)):
        train_fold = ps_meta_array[train_idx]
        valid_fold = ps_meta_array[valid_idx]

        train_dataset = SemanticSegmentationDataset(train_fold)
        valid_dataset = SemanticSegmentationDataset(valid_fold)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=CONFIG['batch_size'], shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=CONFIG['valid_batch_size'], shuffle=False
        )

        # ✅ U-Net with multiclass 설정
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet", 
            in_channels=3,
            classes=len(ss_label_id_list),
            activation=None  # ⛔️ sigmoid 아님
        )

        # ✅ DiceLoss (Multiclass용)
        loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        loss.__name__ = 'DiceLoss'

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, eta_min=1e-6)

        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Fscore(threshold=0.5)
        ]

        train_epoch = smp.utils.train.TrainEpoch(
            model, loss=loss, metrics=metrics, optimizer=optimizer,
            device=CONFIG['device'], verbose=True
        )
        valid_epoch = smp.utils.train.ValidEpoch(
            model, loss=loss, metrics=metrics,
            device=CONFIG['device'], verbose=True
        )

        for epoch in range(CONFIG['num_epochs']):
            print(f'\nEpoch: {epoch}')
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            scheduler.step()

        # 원하는 경우 한 fold만 하고 break
        # break
        
    return model
