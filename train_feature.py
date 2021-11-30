import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse
from scanobjectnn_layer_change_5_best.feature_trainer import ModelNetTrainer
from scanobjectnn_layer_change_5_best.Feature_view_gcn import view_GCN
from scanobjectnn_layer_change_5_best\
    .Feature_dataloader import SinglePoint
def seed_torch(seed=9990):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="view-gcn")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=160)  # it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=0)
parser.add_argument("-lr", type=float, help="learning rate", default=1e-3)
parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training [default: 0.001]')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.09)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-num_views", type=int, help="number of views", default=20)
parser.add_argument("-train_path", type=str, default="/DATA/saber/EXTRACT_FEATURE/SCAN_OBEJCT_NN_FINAL/*/train")
parser.add_argument("-val_path", type=str, default="/DATA/saber/EXTRACT_FEATURE/SCAN_OBEJCT_NN_FINAL/*/test")
# parser.add_argument("--workers", default=torch.get_num_threads())
parser.add_argument("--workers", default=0)
parser.set_defaults(train=False)


def create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)


if __name__ == '__main__':
    seed_torch()
    args = parser.parse_args()
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    n_models_train = args.num_models * args.num_views
    log_dir = args.name + '_stage_2'
    create_folder(log_dir)
    cnet_2 = view_GCN(args.name, nclasses=15, num_views=args.num_views)
    optimizer = optim.Adam(cnet_2.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    train_dataset = SinglePoint(args.train_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.workers,drop_last=True)  # shuffle needs to be false! it's done within the trainer
    val_dataset = SinglePoint(args.val_path, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=args.workers, drop_last=True)
    print('num_train_files: ' + str(len(train_dataset.filepaths)))
    print('num_val_files: ' + str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader,val_loader, optimizer, nn.CrossEntropyLoss(), log_dir, num_views=args.num_views)
    # use trained_view_gcn
    # cnet_2.load_state_dict(torch.load('/DATA/saber/saber_model/model-00001.pth'))
    # trainer.update_validation_accuracy(1)
    trainer.train(50)
