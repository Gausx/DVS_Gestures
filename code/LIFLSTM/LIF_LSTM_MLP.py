"""
LIF-LSTM(MLP) in pytorch. data = DVS-Gestures

Net:MLP(Input-MP4-256FC-11)
"""

import os
import time
import torch
import pandas as pd
import torch.nn as nn
from torch import optim
from module.LIFRNN_module import LIFLSTM
import numpy as np
from DVS_Gesture_dataloders import create_datasets, data_prefetcher
from utils import util
import random
from apex import amp


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    device_ids = range(torch.cuda.device_count())
    print(device_ids)

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

        print('LIF-LSTM(MLP)-DVS-Gesture:dt=', dt, 'ms')

        # Hyper Parameters
        num_epochs = 100  # 100
        batch_size = 72

        batch_size_test = 36
        clip = 10
        is_train_Enhanced = True

        lr = 1e-4
        betas = [0.9, 0.999]
        eps = 1e-8

        T = 60
        ds = 4
        a = 0.25  # hyper-parameters of approximate function
        thresh = 0.3
        lens = a / 2
        decay = 0.3

        target_size = 11  # num_classes
        in_channels = 2  # Green and Red
        im_width, im_height = (128 // ds, 128 // ds)

        # 各种路径的修改
        # 模型保存路径
        modelPath = os.path.dirname(
            os.path.abspath(__file__)) + os.sep + 'results'
        modelNames = 'gesture_LIF-LSTM(MLP)' + str(dt) + 'ms' + '.t7'
        # 运行记录保存路径
        recordPath = os.path.dirname(
            os.path.abspath(__file__)) + os.sep + 'record'
        recordNames = 'gesture_LIF-LSTM_New(MLP)' + str(dt) + 'ms.csv'
        # dataset路径
        savePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))) + os.sep + 'dataset' + os.sep + 'DVS_Gesture' + os.sep  # 保存hdf5路径

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
                                                   num_workers = 16)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size_test,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers = 16)


        # Net
        # fc layer
        cfg_fc = [in_channels * im_width * im_height, 512, target_size]

        # define approximate firing function

        class ActFun(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.gt(thresh).float()

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                temp = abs(input - thresh) < lens
                return grad_input * temp.float() / (2 * lens)

        class Net(nn.Module):

            def __init__(self, ):
                super(Net, self).__init__()

                self.liflstm = LIFLSTM(cfg_fc[0],
                                       cfg_fc[1],
                                       spikeActFun=ActFun.apply,
                                       decay=decay,
                                       onlyLast=False,
                                       dropOut=0.5)

                self.fc = nn.Linear(cfg_fc[1], cfg_fc[2])

            def forward(self, input):
                input = input.reshape(-1,
                                      in_channels * im_width * im_height,
                                      T)

                outputs = self.liflstm(input)
                outputs = torch.sum(outputs, dim=2)
                outputs = self.fc(outputs)

                outputs = outputs / T
                return outputs

        model = Net()
        # model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        print(model)

        weight_decay = 0
        print('weight_decay:', weight_decay)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               betas=betas,
                               eps=eps,
                               weight_decay=weight_decay)

        # 混合精度
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        model = nn.DataParallel(model, device_ids=device_ids)

        # {'Total': 5252619, 'Trainable': 5252619}
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

                input = input.float().permute([0, 2, 3, 4, 1]).to(device)

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
                    input = input_s.reshape(
                        batch_size_test * clip,T,in_channels, im_width, im_height)
                    input = input.float().permute(
                                 [0, 2, 3, 4, 1]).to(device)
                    labels = labels_s.reshape(batch_size_test * clip,T,target_size)
                    labels = labels[:, 1, :].float()

                    outputs = model(input)

                    loss = criterion(outputs.cpu(), labels)

                    _, predicted = torch.max(outputs.data, 1)
                    _, labelTest = torch.max(labels.data, 1)
                    for i in range(batch_size_test):
                        predicted_clips = predicted[i*clip:(i+1)*clip]
                        labelTest_clips = labelTest[i*clip:(i+1)*clip]
                        test_clip_correct = (predicted_clips.cpu()== labelTest_clips).sum()
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

                print('Test Time elasped:', time.time() - start_time)
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

        print('LIF-LSTM(MLP)-DVS-Gesture:dt=', dt, 'ms')
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

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
