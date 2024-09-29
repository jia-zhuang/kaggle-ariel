import os
import sys
import shutil
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from sklearn.model_selection import KFold, StratifiedKFold

from ariel_model import ArielModel


torch.set_float32_matmul_precision('medium')  # 或 'high'
DATA_ROOT = '../input/'
EXP_LOG_ROOT = './experiments/'


def create_dataloader(X_train, Y_train, batch_size=16, shuffle=True):
    X_train = torch.from_numpy(X_train).unsqueeze(1).float()
    Y_train = torch.from_numpy(Y_train).float()
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return training_loader


def train_task(name, ifold, X_train, Y_train, X_val, Y_val):
    train_loader = create_dataloader(X_train, Y_train, shuffle=True)
    val_loader = create_dataloader(X_val, Y_val, shuffle=False)
    
    model = ArielModel(backbone='mobilenet_v3_small')

    # empty output_dir
    output_dir = f'experiments/{name}/{ifold}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir='experiments',
        name=name,
        version=f'{ifold}',
    )

    csv_logger = pl.loggers.CSVLogger(
        save_dir='experiments',
        name=name,
        version=f'{ifold}',
    )

    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        max_epochs=300,
        accelerator='gpu', 
        devices=[ifold % 2],
        log_every_n_steps=len(train_loader),  # log every epoch
        enable_progress_bar=False,
        callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
        # callbacks=[pl.callbacks.RichProgressBar()],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


def prepare_datasets(nfold=4):
    train = np.load('train.npy')
    train_labels = pd.read_csv(f'{DATA_ROOT}/train_labels.csv', index_col='planet_id')
    train_adc_info = pd.read_csv(f'{DATA_ROOT}/train_adc_info.csv', index_col='planet_id')
    kfold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
    indices = list(range(len(train)))
    
    X_train_lst, Y_train_lst, X_val_lst, Y_val_lst = [], [], [], []
    for ifold, (trn_idx, val_idx) in enumerate(kfold.split(indices, y=train_adc_info.star)):
        X_train_lst.append(train[trn_idx])
        Y_train_lst.append(train_labels.values[trn_idx])
        X_val_lst.append(train[val_idx])
        Y_val_lst.append(train_labels.values[val_idx])
    
    return X_train_lst, Y_train_lst, X_val_lst, Y_val_lst


if __name__ == '__main__':
    task_name = sys.argv[1]
    ifold = int(sys.argv[2])
    print('ifold=', ifold)
    datasets = prepare_datasets(nfold=4)
    
    start = time.time()

    # nfold = len(datasets[0])
    # print(f'start {nfold} fold parallel training ...')
    # with ProcessPoolExecutor(2, mp_context=mp.get_context('spawn')) as exe:
    #     results = exe.map(train_task, range(nfold), *datasets)
    # list(results)
    
    train_task(task_name, ifold, *[x[ifold] for x in datasets])
    
    eps = time.time() - start
    print(f'finish training, cost {eps:.3f}s')