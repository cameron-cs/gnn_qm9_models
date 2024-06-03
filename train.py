import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from models import model
from dataset import data
from utils import save_checkpoint, load_checkpoint, load_config, setup_logger
from torch_geometric.datasets import QM9
from torch_geometric.transforms import NormalizeFeatures
import numpy as np


# training and evaluation
def train_model(model, model_alias, optimizer, scheduler, train_loader, val_loader, num_epochs=300, patience=20):
    scaler = GradScaler()
    training_size = len(train_loader.dataset)
    validation_size = len(val_loader.dataset)
    loss_epoch_trn_list, loss_epoch_val_list, MAE_epoch_val_list = [], [], []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        current_epoch = epoch + 1
        running_trn_loss, running_val_loss, error = 0.0, 0.0, 0.0
        logger.info(f"\n [{model_alias}] Epoch {current_epoch}\n-------------------------------")

        model.train()
        for dt in tqdm(train_loader, desc="Training", leave=False):
            dt = dt.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(dt.x, dt.edge_index, dt.batch)
                loss = criterion(pred, dt.y[:, target])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_trn_loss += loss.item()

        epoch_mean_trn_loss = running_trn_loss / training_size
        loss_epoch_trn_list.append(epoch_mean_trn_loss)
        logger.info(f"[{model_alias}] Training MAE loss: {loss_epoch_trn_list[-1]:.7f}")

        model.eval()
        with torch.no_grad():
            for dv in tqdm(val_loader, desc="Evaluating", leave=False):
                dv = dv.to(device)
                pred = model(dv.x, dv.edge_index, dv.batch)
                val_loss = criterion(pred, dv.y[:, target])
                running_val_loss += val_loss.item()
                error += (pred * tr_std + tr_mean - dv.y[:, target]).abs().sum().item()

        epoch_mean_val_loss = running_val_loss / validation_size
        MAE_epoch_val_list.append(error / len(val_dataset))
        loss_epoch_val_list.append(epoch_mean_val_loss)
        logger.info(f"[{model_alias}] Validation MAE: {loss_epoch_val_list[-1]:.7f}")
        logger.info(f"[{model_alias}] MAE in validation set: {error / len(val_dataset):.7f}")

        scheduler.step(epoch_mean_val_loss)

        # early stopping
        if epoch_mean_val_loss < best_val_loss:
            best_val_loss = epoch_mean_val_loss
            patience_counter = 0
            logger.info("Model saved\n")
            save_checkpoint({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_mean_val_loss,
            }, filename=f'checkpoint/{model_alias}_best_model_epoch_{current_epoch}.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {current_epoch}")
            break

    return loss_epoch_trn_list, loss_epoch_val_list, MAE_epoch_val_list


if __name__ == '__main__':
    # load configuration
    config = load_config('config.yaml')

    # setup logger
    logger = setup_logger('training.log')

    # data loading and preprocessing
    path = config['data']['path']
    dataset = QM9(path, transform=NormalizeFeatures())

    indices = np.random.RandomState(seed=42).permutation(len(dataset))

    # split datasets
    batch_size = config['training']['batch_size']
    target = 15

    test_dataset = dataset[indices[0:10000]]
    val_dataset = dataset[indices[10000:20000]]
    train_dataset = dataset[indices[20000:]]

    # normalize target variable
    tr_sum = sum([data.y[:, target] for data in train_dataset])
    tr_mean = (tr_sum / len(train_dataset)).item()

    std_sum = sum([(data.y[:, target] - tr_mean) ** 2 for data in train_dataset])
    tr_std = np.sqrt(std_sum / len(train_dataset)).item()

    for data in train_dataset:
        data.y[:, target] = (data.y[:, target] - tr_mean) / tr_std

    # data loaders
    data_loader_manager = data.data_loader_manager.DataLoaderManager(train_dataset, val_dataset, test_dataset, batch_size=batch_size,
                                                                     num_workers=config['training']['num_workers'])

    train_loader = data_loader_manager.get_train_loader()
    val_loader = data_loader_manager.get_val_loader()
    test_loader = data_loader_manager.get_test_loader()

    # the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.GraphClassificationModel(num_features=dataset.num_features, hidden_channels=config['model']['hidden_channels']).to(device)

    # optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode=config['scheduler']['mode'], factor=config['scheduler']['factor'],
                                  patience=config['scheduler']['patience'], min_lr=config['scheduler']['min_lr'])
    criterion = torch.nn.L1Loss()

    # training the model
    gcn_train_loss, gcn_val_loss, gcn_val_mae = train_model(model, 'GraphClassificationModel', optimizer, scheduler, train_loader,
                                                            val_loader,
                                                            num_epochs=config['training']['num_epochs'],
                                                            patience=config['training']['patience'])
