"""
LSTM(Rate-Coding-inspired)(CNN) in pytorch. data = DVS-Gestures

Net:CNN(Input-MP4-64C3-128C3-AP2-128C3-AP2-256FC-11)
"""
import os
import pandas as pd
import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import util
import random
from module import convGru
import numpy as np
from apex import amp
from DVS_Gesture_dataloders import create_datasets

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    device_ids = range(torch.cuda.device_count())
    print(device_ids)

    # m = [1, 5, 10, 15, 20, 25]
    m = [25]

    for dt in m:
        # 随机种子
        seed = 666
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print('GRU(Rate-Coding-inspired)(CNN)-DVS-Gestures:dt=', dt, 'ms')
        # Hyper Parameters
        num_epochs = 100  # 100
        batch_size = 36

        batch_size_test = 36
        clip = 1
        is_train_Enhanced = False

        lr = 1e-4
        betas = [0.9, 0.999]
        eps = 1e-8

        T = 60
        ds = 4

        target_size = 11  # num_classes
        in_channels = 2  # Green and Red
        im_width, im_height = (128 // ds, 128 // ds)

        # 各种路径的修改
        # 模型保存路径
        modelPath = os.path.dirname(
            os.path.abspath(__file__)) + os.sep + 'results'
        modelNames = 'gestures_gru(Rate-Coding-inspired)(CNN)' + \
            str(dt) + 'ms' + '.t7'
        # 运行记录保存路径
        recordPath = os.path.dirname(
            os.path.abspath(__file__)) + os.sep + 'record'
        recordNames = 'gestures_gru(Rate-Coding-inspired)(CNN)' + \
            str(dt) + 'ms.csv'
        # dataset路径
        savePath = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))) + os.sep + 'dataset' + os.sep + 'DVS_Gesture' + os.sep  # 保存hdf5路径
        # print(savePath)


        # Data set
        train_dataset = create_datasets(savePath,
                                        train=True,
                                        is_train_Enhanced=is_train_Enhanced,
                                        ds=ds,
                                        dt=dt * 1000,
                                        chunk_size_train=T,
                                        )

        test_dataset = create_datasets(savePath,
                                       train=False,
                                       ds=ds,
                                       dt=dt * 1000,
                                       chunk_size_test=T,
                                       clip=clip
                                       )
        # Data loader
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=False,
                                                   num_workers = 4)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size_test,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers = 4)
        # Net
        # cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
        cfg_cnn = [(2, 64, 1, 1, 3), (64, 128, 1, 1, 3), (128, 128, 1, 1, 3)]
        # pooling kernel_size
        cfg_pool = [4, 2, 2]
        # fc layer
        cfg_fc = [cfg_cnn[2][1] * 8 * 8, 256, target_size]

        class Net(nn.Module):
            def __init__(self, ):
                super(Net, self).__init__()
                in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
                self.convgru1 = convGru.ConvGRU(
                    input_size=(im_width, im_height),
                    input_dim=in_planes,
                    hidden_dim=out_planes,
                    kernel_size=(kernel_size, kernel_size),
                    batch_first=True,
                    num_layers=1
                )

                in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
                self.convgru2 = convGru.ConvGRU(
                    input_size=(im_width, im_height),
                    input_dim=in_planes,
                    hidden_dim=out_planes,
                    kernel_size=(kernel_size, kernel_size),
                    batch_first=True,
                    num_layers=1
                )
                kernel_size = cfg_pool[1]
                self.avgPool1 = nn.AvgPool3d(
                    kernel_size=(1, kernel_size, kernel_size))

                in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
                self.convgru3 = convGru.ConvGRU(
                    input_size=(im_width // 2, im_height // 2),
                    input_dim=in_planes,
                    hidden_dim=out_planes,
                    kernel_size=(kernel_size, kernel_size),
                    batch_first=True,
                    num_layers=1
                )
                kernel_size = cfg_pool[2]
                self.avgPool2 = nn.AvgPool3d(
                    kernel_size=(1, kernel_size, kernel_size))

                self.gru = nn.GRU(
                    input_size=cfg_fc[0], hidden_size=cfg_fc[1], num_layers=2
                )

                self.linear = nn.Linear(cfg_fc[1], cfg_fc[2])

            def forward(self, input, hidden_prev=None):
                torch.cuda.empty_cache()
                if hidden_prev is None:
                    hidden_prev = []

                x = input
                del input
                torch.cuda.empty_cache()
                x, _ = self.convgru1(x)
                x = x[0]
                x = F.dropout(x, p=0.5)
                torch.cuda.empty_cache()

                x, _ = self.convgru2(x)
                x = x[0]
                x = self.avgPool1(x)
                x = F.dropout(x, p=0.5)

                x, _ = self.convgru3(x)
                x = x[0]
                x = self.avgPool2(x)
                x = x.reshape(-1, T, cfg_fc[0])
                x = F.dropout(x, p=0.5)
                if not hasattr(self, '_flattened'):
                    self.gru.flatten_parameters()
                setattr(self, '_flattened', True)
                r_out, _ = self.gru(x)
                r_out = F.dropout(r_out, p=0.5)
                out = torch.mean(self.linear(r_out), 1)

                del x, r_out
                torch.cuda.empty_cache()
                return out

        model = Net()
        # model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        print(model)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)

        # 混合精度
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        model = nn.DataParallel(model, device_ids=device_ids)


        print(util.get_parameter_number(model))


        best_acc = 0
        best_epoch = 0
        acc_record = list([])

        epoch_list = []
        loss_train_list = []
        loss_test_list = []
        acc_train_list = []
        acc_test_list = []

        for epoch in range(num_epochs):

            model.train()
            start_time = time.time()
            running_loss = 0
            train_correct = 0
            train_total = 0
            for batch_idx, (input, labels) in enumerate(train_loader):
                model.zero_grad()
                optimizer.zero_grad()

                input = input.float().to(device)

                labels = labels[:, 1, :].float()

                outputs = model(input)

                loss = criterion(outputs.cpu(), labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                _, labelTest = torch.max(labels.data, 1)
                train_total += labelTest.size(0)
                train_correct += (predicted.cpu() == labelTest).sum()

                # backprop
                # loss.backward()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                optimizer.step()

                print('train: Epoch [%d/%d], Step [%d/%d], Loss: %.5f' %
                      (epoch + 1,
                       num_epochs,
                       batch_idx + 1,
                       len(train_loader),
                       loss.item()))

            print('Train Time elasped:', time.time() - start_time)

            running_loss = running_loss / len(train_loader)
            epoch_list.append(epoch + 1)
            train_acc = 100. * float(train_correct) / float(train_total)

            print('Train datasets total:', train_total)
            print('Tarin loss:%.5f' % running_loss)
            print('Train acc: %.3f' % train_acc)

            loss_train_list.append(running_loss)
            acc_train_list.append(train_acc)

            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            start_time = time.time()
            # test
            with torch.no_grad():

                for batch_idx, (input_s, labels_s) in enumerate(test_loader):


                    for i in range(batch_size_test):
                        input = input_s[i]
                        input = input.float().to(device)

                        labels = labels_s[i]
                        labels = labels[:, 1, :].float()

                        outputs = model(input)

                        loss = criterion(outputs.cpu(), labels)


                        _, predicted = torch.max(outputs.data, 1)
                        _, labelTest = torch.max(labels.data, 1)
                        test_clip_correct = (predicted.cpu()
                                              == labelTest).sum()

                        if test_clip_correct.item() / clip > 0.5:
                            test_correct += 1
                        test_total += 1
                        test_loss += loss / clip

                    print('test: Epoch [%d/%d], Step [%d/%d], Loss: %.5f' %
                          (epoch + 1,
                           num_epochs,
                           batch_idx + 1,
                           len(test_loader),
                           loss.item()))

                test_loss = test_loss / test_total
                test_acc = 100. * float(test_correct) / float(test_total)
                loss_test_list.append(test_loss)

                print('Train Time elasped:', time.time() - start_time)
                print('Test datasets total:', test_total)
                print('Test loss:%.5f' % test_loss)
                print('Test acc: %.3f' % test_acc)

                acc_test_list.append(test_acc)
                acc_record.append(test_acc)

                # print('Saving..')
                # state = {'net': model.state_dict(),
                #          'acc': test_acc,
                #          'epoch': (epoch + 1) // n_test_interval,
                #          'acc_record': acc_record,
                #          }

                if test_acc > best_acc:
                    best_epoch = epoch + 1
                    best_acc = test_acc

                    # if not os.path.exists(modelPath):
                    #     os.makedirs(modelPath)
                    # torch.save(state, modelPath + os.sep + modelNames)

        print('GRU(Rate-Coding-inspired)(CNN)-DVS-Gestures:dt=', dt, 'ms')
        print('best acc:', best_acc, 'best_epoch:', best_epoch)
        epoch_list.append(best_epoch)
        acc_test_list.append(best_acc)

        lists = [loss_train_list,
                 loss_test_list,
                 acc_train_list,
                 acc_test_list]
        test = pd.DataFrame(data=lists,
                            index=['Train_Loss',
                                   'Test_Loss',
                                   'Train_Accuracy',
                                   'Test_Accuracy'],
                            columns=epoch_list)
        test.index.name = 'Epochs'

        if not os.path.exists(recordPath):
            os.makedirs(recordPath)
        test.to_csv(recordPath + os.sep + recordNames)


if __name__ == '__main__':
    main()
