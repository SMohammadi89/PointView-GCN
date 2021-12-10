from dataloader import SinglePoint
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """ PARAMETERS """
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=54, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='POINTNET_MSG_CLS', help='model name [default: POINTNET3_MSG_CLS]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--ngpu', type=int, default=[0, 1, 2], help='specify how many gpus to train the model [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--test_path', default='../data/single_view_modelnet/*/test', help='path of the test data')
    parser.add_argument('--train_path', default='../data/single_view_modelnet/*/train', help='path of the train data')
    parser.add_argument("--worker", type=int, default=torch.get_num_threads())
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for points, target in tqdm(loader):
        points = points.transpose(2, 1)
        points, target = points.float().to(device), target.to(device)
        classifier = model.eval()
        vote_pool = torch.zeros(target.shape[0], num_class).to(device)
        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]
        for cat in torch.unique(target):
            classacc = pred_choice[target == cat].eq(target[target == cat]).cpu().sum()
            class_acc[cat, 0] += classacc.item() / points[target == cat].size()[0]
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target).cpu().sum()
        batch_num_samples = points.size()[0]
        mean_correct.append(correct.item() / batch_num_samples)
    class_acc[:, 2] = np.nan_to_num(class_acc[:, 0] / class_acc[:, 1])
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    """ CREATE DIR """
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    """ LOG """
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    """ DATA LOADING """
    log_string('Load dataset ...')
    TRAIN_DATASET = SinglePoint(args.train_path, npoint=args.num_point)
    TEST_DATASET = SinglePoint(args.test_path, npoint=args.num_point)
    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=54, shuffle=True, num_workers=0)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=30, shuffle=False, num_workers=0)
    print('num_train_files: ' + str(len(TRAIN_DATASET.filepaths)))
    print('num_val_files: ' + str(len(TEST_DATASET.filepaths)))

   
    """ MODEL LOADING """
    num_class = 40
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))
    #train on the single gpu
    #classifier = MODEL.get_model(num_class).to(device)
    # train on the multiple gpu
    classifier = nn.DataParallel(MODEL.get_model(num_class).to(device), device_ids=args.ngpu)
    criterion = MODEL.get_loss().to(device)

    try:
        checkpoint = torch.load('')
        start_epoch = checkpoint['epoch']
        classifier.module.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0 
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(),
                                     lr=args.learning_rate,
                                     betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    """ TRAINING """
    logger.info('Start training...')
    train_instance_acc, instance_acc, class_acc = [], [], []
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        scheduler.step()
        for points, target in tqdm(trainDataLoader):
            points = points.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.tensor(points).float()
            points = points.transpose(2, 1)
            # target = target.unsqueeze(1).repeat(1, 4 * (5)).view(-1)
            points, target = points.to(device), target.long().to(device)
            optimizer.zero_grad()
            batch_num_samples = points.size()[0]
            classifier = classifier.train()
            pred, trans_feat = classifier(points.float())
            loss = criterion(pred, target, trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target).cpu().sum()
            correct_ = correct.item() / batch_num_samples
            mean_correct.append(correct_)
            loss.mean().backward()
            optimizer.step()
            global_step += 1
        _train_instance_acc = np.mean(mean_correct)
        with torch.no_grad():
            _instance_acc, _class_acc = test(classifier, testDataLoader)
            if _instance_acc >= best_instance_acc:
                best_instance_acc = _instance_acc
                best_epoch = epoch + 1
            if _class_acc >= best_class_acc:
                best_class_acc = _class_acc
            if _instance_acc >= best_instance_acc:
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {'epoch': best_epoch,
                         'instance_acc': _instance_acc,
                         'class_acc': _class_acc,
                         'model_state_dict': classifier.module.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict()}
                torch.save(state, savepath)
            global_epoch += 1
        log_string('Train Instance Accuracy: %f' % _train_instance_acc)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (_instance_acc, _class_acc))
        log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
        train_instance_acc.append(_train_instance_acc)
        instance_acc.append(_instance_acc)
        class_acc.append(_class_acc)
    logger.info('End of training...')

    """ PLOTTING """
    plt.plot(train_instance_acc, label='Train Accuracy')
    plt.plot(instance_acc, label='Test Accuracy')
    plt.title('Model Accuracy ({} epochs)'.format(args.epoch))
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
