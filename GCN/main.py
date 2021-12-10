import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import math
from PintView_GCN import PointViewGCN
from dataloader import MultiviewPoint

def seed_torch(seed=9990):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
class Trainer(object):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn,log_dir, num_views=20):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.num_views = num_views
        self.model.cuda()
        self.log_dir = log_dir
        self.writer = SummaryWriter()

    def train(self, n_epochs):
        global lr
        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):
            if epoch == 1:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                if epoch > 1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5 * (1 + math.cos(epoch * math.pi / 15))

            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths) / self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[
                                     rand_idx[i] * self.num_views:(rand_idx[i] + 1) * self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            for i, data in enumerate(self.train_loader):
                if epoch == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr * ((i + 1) / (len(rand_idx) // 20))

                B, V, C = data[1].size()
                in_data = Variable(data[1]).view(-1, C)
                target = Variable(data[0]).cuda().long()
                target_ = target.unsqueeze(1).repeat(1, 4 * (5 + 10+15)).view(-1)
                self.optimizer.zero_grad()
                out_data, F_score, F_score_m, F_score2 = self.model(in_data)
                out_data_ = torch.cat((F_score, F_score_m, F_score2), 1).view(-1, 40)
                loss = self.loss_fn(out_data, target) + self.loss_fn(out_data_, target_)


                self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)
                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)
                loss.backward()
                self.optimizer.step()

                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch + 1, i + 1, loss, acc)
                if (i + 1) % 1 == 0:
                    print(log_str)
            i_acc += i
            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                # save best model
                if val_overall_acc > best_acc:
                    best_acc = val_overall_acc
                    print('the best model_is saving')
                    self.model.save(self.log_dir + '/')
                print('best_acc', best_acc)
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0
        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        self.model.eval()
        targ_numpy, pred_numpy = [], []
        for _, data in enumerate(self.val_loader, 0):
            B, V, C = data[1].size()
            in_data = Variable(data[1]).view(-1, C)
            target = Variable(data[0]).cuda()
            out_data, F_score, F_score_m, F_score2 = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            targ_numpy.append(np.asarray(target.cpu()))
            pred_numpy.append(np.asarray(pred.cpu()))
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]
        print('Total # of test models: ', all_points)
        class_acc = (samples_class - wrong_class) / samples_class
        val_mean_class_acc = np.mean(class_acc)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print(class_acc)
        self.model.train()
        return val_overall_acc, val_mean_class_acc


parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="path of the model", default="log")
parser.add_argument("-bs", "--batchSize", type=int, default=128)
parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training [default: 0.0001]')
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.09)
parser.add_argument("-num_views", type=int, help="number of views", default=20)
parser.add_argument('--train_path', default='../data/modelnet_trained_feature/*/train', help='path of the trained feature for train data')
parser.add_argument('--val_path', default='../data/modelnet_trained_feature/*/test', help='path of the trained feature for test data')
parser.add_argument("--workers", default=0)
parser.set_defaults(train=False)

if __name__ == '__main__':
    seed_torch()
    args = parser.parse_args()
    log_dir = args.name
    cnet_2 = PointViewGCN(args.name, nclasses=40, num_views=args.num_views)
    optimizer = optim.Adam(cnet_2.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    train_dataset = MultiviewPoint(args.train_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.workers)
    val_dataset = MultiviewPoint(args.val_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=args.workers)
    print('num_train_files: ' + str(len(train_dataset.filepaths)))
    print('num_val_files: ' + str(len(val_dataset.filepaths)))
    trainer = Trainer(cnet_2, train_loader,val_loader, optimizer, nn.CrossEntropyLoss(),log_dir, num_views=args.num_views)
    # cnet_2.load_state_dict(torch.load('path of the pre-trained model'))
    # trainer.update_validation_accuracy(1)
    trainer.train(100)
