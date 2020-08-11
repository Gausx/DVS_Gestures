"""
SNNs(CNN) in pytorch. data = DVS-Gestures

Net:CNN(Input-MP4-64C3-128C3-AP2-128C3-AP2-256FC-11)
"""


import torch
import pandas as pd
import time
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils import util
import random
from apex import amp
import numpy as np
from DVS_Gesture_dataloders import create_datasets, data_prefetcher
import os


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

        print('SNNs(CNN)-DVS-Gestures:dt=', dt, 'ms')

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
        a = 0.5  # hyper-parameters of approximate function
        lens = a / 2
        decay = 0.3

        target_size = 11  # num_classes
        in_channels = 2  # Green and Red
        thresh = 0.3
        im_width, im_height = (128 // ds, 128 // ds)

        # 各种路径的修改
        # 模型保存路径
        modelPath = os.path.dirname(
            os.path.abspath(__file__)) + os.sep + 'results'
        modelNames = 'gestures_snn(CNN)' + str(dt) + 'ms' + '.t7'
        # 运行记录保存路径
        recordPath = os.path.dirname(
            os.path.abspath(__file__)) + os.sep + 'record'
        recordNames = 'gestures_snn(CNN)' + str(dt) + 'ms.csv'
        # dataset路径
        savePath = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))))+ os.sep + 'dataset' + os.sep + 'DVS_Gesture' + os.sep  # 保存hdf5路径

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
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size_test,
                                                  shuffle=False,
                                                  drop_last=False)

        # Net

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

        act_fun = ActFun.apply

        # cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
        cfg_cnn = [(2, 64, 1, 1, 3),
                   (64, 128, 1, 1, 3),
                   (128, 128, 1, 1, 3)]
        # kernel size
        cfg_kernel = [128, 64, 8]
        # fc layer
        cfg_fc = [256, target_size]

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
                self.conv1 = nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
                in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
                self.conv2 = nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
                in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
                self.conv3 = nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)

                self.fc1 = nn.Linear(
                    cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
                self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

            def forward(self, input):
                b, _, _, _, time_window = input.size()
                c1_mem = c1_spike = torch.zeros(
                    b, cfg_cnn[0][1], 32, 32, device=device)
                c2_mem = c2_spike = torch.zeros(
                    b, cfg_cnn[1][1], 32, 32, device=device)
                c3_mem = c3_spike = torch.zeros(
                    b, cfg_cnn[2][1], 16, 16, device=device)

                p1_mem = p1_spike = torch.zeros(
                    b, cfg_cnn[1][1], 16, 16, device=device)
                p2_mem = p2_spike = torch.zeros(
                    b, cfg_cnn[2][1], 8, 8, device=device)
                h1_mem = h1_spike = h1_sumspike = torch.zeros(
                    b, cfg_fc[0], device=device)
                h2_mem = h2_spike = h2_sumspike = torch.zeros(
                    b, cfg_fc[1], device=device)

                for step in range(time_window):  # simulation time steps
                    x = input[:, :, :, :, step].to(device)
                    c1_mem, c1_spike = mem_update(
                        self.conv1, x, c1_mem, c1_spike)
                    c1_spike = F.dropout(c1_spike, p=0.5)
                    torch.cuda.empty_cache()

                    c2_mem, c2_spike = mem_update(
                        self.conv2, c1_spike, c2_mem, c2_spike)
                    p1_mem, p1_spike = mem_update_pool(
                        F.avg_pool2d, c2_spike, p1_mem, p1_spike)
                    p1_spike = F.dropout(p1_spike, p=0.5)

                    c3_mem, c3_spike = mem_update(
                        self.conv3, p1_spike, c3_mem, c3_spike)
                    p2_mem, p2_spike = mem_update_pool(
                        F.avg_pool2d, c3_spike, p2_mem, p2_spike)
                    b, _, _, _ = p2_spike.size()
                    x = p2_spike.view(b, -1)
                    x = F.dropout(x, p=0.5)
                    h1_mem, h1_spike = mem_update(
                        self.fc1, x, h1_mem, h1_spike)
                    h1_sumspike += h1_spike
                    h1_spike = F.dropout(h1_spike, p=0.5)

                    h2_mem, h2_spike = mem_update(
                        self.fc2, h1_spike, h2_mem, h2_spike)
                    h2_sumspike += h2_spike

                    del x
                    torch.cuda.empty_cache()

                outputs = h2_sumspike / time_window
                return outputs

        def mem_update(fc, x, mem, spike):
            torch.cuda.empty_cache()
            mem = mem * decay * (1 - spike) + fc(x)
            spike = act_fun(mem)
            return mem, spike

        def mem_update_pool(opts, x, mem, spike, pool=2):
            torch.cuda.empty_cache()
            mem = mem * decay * (1 - spike) + opts(x, pool)
            spike = act_fun(mem)
            return mem, spike

        model = Net().to(device)
        print(model)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)

        # 混合精度
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        model = nn.DataParallel(model, device_ids=device_ids)

        # {'Total': 2322891, 'Trainable': 2322891}
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

                    for i in range(batch_size_test):
                        input = input_s[i]
                        input = input.float().permute(
                            [0, 2, 3, 4, 1]).to(device)

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

        print('SNNs(CNN)-DVS-Gestures:dt=', dt, 'ms')
        print('best acc:', best_acc, 'best_epoch:', best_epoch)

        epoch_list.append(best_epoch)
        acc_test_list.append(best_acc)

        lists = [loss_train_list,
                 loss_test_list,
                 acc_train_list,
                 acc_test_list]
        test = pd.DataFrame(
            data=lists,
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
