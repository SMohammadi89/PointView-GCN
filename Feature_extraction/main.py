#Here we extract feature using pre-trained pointnet++ model on the single-view pcd
#The extracted shape feature would be thee input of the GCN for shape classification

from dataloader import SinglePoint
import numpy as np
import os
import torch
from tqdm import tqdm
import pointnet2_cls as pointnet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batchSize", type=int, default=1)
parser.add_argument("-num_class", type=int, default=40)
parser.add_argument('--val_path', default='../data/single_view_modelnet/modelnetdata/*/test', help='path of the test data')
parser.add_argument('--train_path', default='../data/single_view_modelnet/modelnetdata/*/train', help='path of the train data')
parser.add_argument('--output_data_path', default='../data/modelnet_trained_feature/', help='path of the output feature')
parser.add_argument('--checkpoint_path', default='../log/pointnet_on_single_view.pth', help='path of the pre_trained model')
parser.add_argument("--workers", default=0)
parser.set_defaults(train=False)


if __name__ == '__main__':
    args = parser.parse_args()
    train_dataset = SinglePoint(args.train_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False,batch_size=args.batchSize) # shuffle needs to be false! it's done within the trainer
    val_dataset = SinglePoint(args.val_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,batch_size=args.batchSize)

    model = pointnet.get_model(args.num_class).cuda()
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    mean_correct_train = []
    class_acc_train = np.zeros((args.num_class, 3))
    mean_correct_test = []
    class_acc_test = np.zeros((args.num_class, 3))
    with torch.no_grad():
        for j, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            points = np.asarray(data[0], dtype=np.float32)
            target1 = data[1]
            points = points.transpose(0, 2, 1)
            points = torch.tensor(points)
            points, target1 = points.cuda(), target1.cuda()
            classifier = model.eval()
            _, _, features_train = classifier.forward(points)
            name = data[3][-1]
            name1 = os.path.split(name)[-1]
            # scene_name = name1[:-13]
            scene_name = data[2][-1]
            name2 = name1[:-4]
            out_file = args.output_data_path + str(scene_name) + '/' + 'train'
            if not os.path.exists(out_file):
                os.makedirs(out_file)

            path = os.path.join(out_file, name2 + '.pth')
            torch.save(features_train, path)
            vote_pool = torch.zeros(target1.shape[0], args.num_class).cuda()
            for _ in range(1):
                pred, _,_  = classifier(points)
                vote_pool += pred
            pred_choice = pred.data.max(1)[1]
            for cat in np.unique(target1.cpu()):
                classacc = pred_choice[target1 == cat].eq(target1[target1 == cat].long().data).cpu().sum()
                class_acc_train[cat, 0] += classacc.item() / float(points[target1 == cat].size()[0])
                class_acc_train[cat, 1] += 1
            correct = pred_choice.eq(target1.long().data).cpu().sum()
            mean_correct_train.append(correct.item() / float(points.size()[0]))
        class_acc_train[:, 2] = class_acc_train[:, 0] / class_acc_train[:, 1]
        print(class_acc_train[:, 2])
        class_acc_train = np.mean(class_acc_train[:, 2])
        instance_acc_train = np.mean(mean_correct_train)
        print('Train Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc_train, class_acc_train))

        for j, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            points = np.asarray(data[0], dtype=np.float32)
            target1 = data[1]
            points = points.transpose(0, 2, 1)
            points = torch.tensor(points)
            points, target1 = points.cuda(), target1.cuda()
            classifier = model.eval()
            _, _, features_test = classifier.forward(points)

            name = data[3][-1]
            name1 = os.path.split(name)[-1]
            # scene_name = name1[:-13]
            scene_name = data[2][-1]
            name2 = name1[:-4]
            out_file = args.output_data_path + str(scene_name) + '/' + 'test'
            if not os.path.exists(out_file):
                os.makedirs(out_file)

            path = os.path.join(out_file, name2 + '.pth')
            torch.save(features_test, path)
            vote_pool = torch.zeros(target1.shape[0], args.num_class).cuda()
            for _ in range(1):
                pred, _,_  = classifier(points)
                vote_pool += pred
            pred_choice = pred.data.max(1)[1]
            for cat in np.unique(target1.cpu()):
                classacc = pred_choice[target1 == cat].eq(target1[target1 == cat].long().data).cpu().sum()
                class_acc_test[cat, 0] += classacc.item() / float(points[target1 == cat].size()[0])
                class_acc_test[cat, 1] += 1
            correct = pred_choice.eq(target1.long().data).cpu().sum()
            mean_correct_test.append(correct.item() / float(points.size()[0]))
        class_acc_test[:, 2] = class_acc_test[:, 0] / class_acc_test[:, 1]
        print(class_acc_test[:, 2])
        class_acc_test = np.mean(class_acc_test[:, 2])
        instance_acc_test = np.mean(mean_correct_test)
        print('test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc_test, class_acc_test))






















