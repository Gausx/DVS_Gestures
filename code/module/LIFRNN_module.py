import torch
from torch import nn
from module.LIFRNN import *

torch.manual_seed(666)  # 随机种子
torch.cuda.manual_seed_all(666)


class LIFRNN(nn.Module):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 spikeActFun,
                 decay=0.3,
                 onlyLast=False,
                 dropOut=0,
                 useBatchNorm=False,
                 useLayerNorm=False
                 ):
        super().__init__()
        self.onlyLast = onlyLast
        self.network = nn.Sequential()
        self.network.add_module('LIF-RNN',
                                LIFRNNCell(inputSize,
                                           hiddenSize,
                                           spikeActFun,
                                           decay=decay,
                                           dropOut=dropOut,
                                           useBatchNorm=useBatchNorm,
                                           useLayerNorm=useLayerNorm))

    def forward(self, data):
        torch.cuda.empty_cache()
        for layer in self.network:
            layer.reset()
        for step in range(list(data.size())[-1]):
            out = data[:, :, step]
            for layer in self.network:
                out = layer(out)
            output = out

            if step == 0:
                temp = list(output.size())
                temp.append(data.size()[-1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)
            outputsum[:, :, step] = output

        if self.onlyLast:
            return output
        else:
            return outputsum

# 确认LIF-LSTM中的输入参数
#   todo


class LIFLSTM(nn.Module):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 spikeActFun,
                 decay=0.3,
                 onlyLast=False,
                 dropOut=0,
                 useBatchNorm=False,
                 useLayerNorm=False,):
        super().__init__()
        self.onlyLast = onlyLast
        self.network = nn.Sequential()
        self.network.add_module('LIF-LSTM',
                                LIFLSTMCell(inputSize,
                                            hiddenSize,
                                            spikeActFun,
                                            decay=decay,
                                            dropOut=dropOut,
                                            useBatchNorm=useBatchNorm,
                                            useLayerNorm=useLayerNorm,
                                            ))

    def forward(self, data):

        torch.cuda.empty_cache()

        for layer in self.network:
            layer.reset()

        for step in range(list(data.size())[-1]):
            out = data[:, :, step]
            for layer in self.network:
                out = layer(out)
            output = out

            if step == 0:
                temp = list(output.size())
                temp.append(data.size()[-1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)
            outputsum[:, :, step] = output

        if self.onlyLast:
            return output
        else:
            return outputsum


class LIFGRU(nn.Module):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 spikeActFun,
                 decay=0.3,
                 onlyLast=False,
                 dropOut=0,
                 useBatchNorm=False,
                 useLayerNorm=False,):
        super().__init__()
        self.onlyLast = onlyLast
        self.network = nn.Sequential()
        self.network.add_module('LIF-GRU',
                                LIFGRUCell(inputSize,
                                           hiddenSize,
                                           spikeActFun,
                                           decay=decay,
                                           dropOut=dropOut,
                                           useBatchNorm=useBatchNorm,
                                           useLayerNorm=useLayerNorm,
                                           ))

    def forward(self, data):
        torch.cuda.empty_cache()
        for layer in self.network:
            layer.reset()
        for step in range(list(data.size())[-1]):
            out = data[:, :, step]
            for layer in self.network:
                out = layer(out)
            output = out

            if step == 0:
                temp = list(output.size())
                temp.append(data.size()[-1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)
            outputsum[:, :, step] = output

        if self.onlyLast:
            return output
        else:
            return outputsum


class LIFConvRNN(nn.Module):
    def __init__(self,
                 input_range,
                 inputSize,
                 hiddenSize,
                 kernel_size,
                 spikeActFun,
                 bias=True,
                 decay=0.3,
                 onlyLast=False,
                 dropOut=0,
                 useBatchNorm=False,
                 useLayerNorm=False,):
        super().__init__()

        self.onlyLast = onlyLast

        self.network = nn.Sequential()
        self.network.add_module('LIFConvRNN',
                                LIFConvRNNCell(input_range=input_range,
                                               inputSize=inputSize,
                                               hiddenSize=hiddenSize,
                                               kernel_size=kernel_size,
                                               bias=bias,
                                               spikeActFun=spikeActFun,
                                               dropOut=dropOut,
                                               useBatchNorm=useBatchNorm,
                                               useLayerNorm=useLayerNorm,
                                               decay=decay))

    def forward(self, data, hidden_state=None):

        torch.cuda.empty_cache()

        for layer in self.network:
            layer.reset()

        for step in range(list(data.size())[-1]):
            out = data[:, :, :, :, step]
            for layer in self.network:
                out = layer(out)
            output = out

            if step == 0:
                temp = list(output.size())
                temp.append(data.size()[-1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)

            outputsum[:, :, :, :, step] = output

        if self.onlyLast:
            return output
        else:
            return outputsum


class LIFConvLSTM(nn.Module):
    def __init__(self,
                 input_range,
                 inputSize,
                 hiddenSize,
                 kernel_size,
                 spikeActFun,
                 bias=True,
                 decay=0.3,
                 onlyLast=False,
                 dropOut=0,
                 useBatchNorm=False,
                 useLayerNorm=False,):
        super().__init__()

        self.onlyLast = onlyLast

        self.network = nn.Sequential()
        self.network.add_module('LIFConvLSTM',
                                LIFConvLSTMCell(input_range=input_range,
                                                inputSize=inputSize,
                                                hiddenSize=hiddenSize,
                                                kernel_size=kernel_size,
                                                bias=bias,
                                                spikeActFun=spikeActFun,
                                                dropOut=dropOut,
                                                useBatchNorm=useBatchNorm,
                                                useLayerNorm=useLayerNorm,
                                                decay=decay))

    def forward(self, data, hidden_state=None):

        torch.cuda.empty_cache()

        for layer in self.network:
            layer.reset()

        for step in range(list(data.size())[-1]):
            out = data[:, :, :, :, step]
            for layer in self.network:
                out = layer(out)
            output = out

            if step == 0:
                temp = list(output.size())
                temp.append(data.size()[-1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)

            outputsum[:, :, :, :, step] = output

        if self.onlyLast:
            return output
        else:
            return outputsum


class LIFConvGRU(nn.Module):
    def __init__(self,
                 input_range,
                 inputSize,
                 hiddenSize,
                 kernel_size,
                 spikeActFun,
                 bias=True,
                 decay=0.3,
                 onlyLast=False,
                 dropOut=0,
                 useBatchNorm=False,
                 useLayerNorm=False,):
        super().__init__()

        self.onlyLast = onlyLast

        self.network = nn.Sequential()
        self.network.add_module('LIFConvGRU',
                                LIFConvGRUCell(input_range=input_range,
                                               inputSize=inputSize,
                                               hiddenSize=hiddenSize,
                                               kernel_size=kernel_size,
                                               bias=bias,
                                               spikeActFun=spikeActFun,
                                               dropOut=dropOut,
                                               useBatchNorm=useBatchNorm,
                                               useLayerNorm=useLayerNorm,
                                               decay=decay
                                               ))

    def forward(self, data, hidden_state=None):

        torch.cuda.empty_cache()

        for layer in self.network:
            layer.reset()

        for step in range(list(data.size())[-1]):

            out = data[:, :, :, :, step]

            for layer in self.network:
                out = layer(out)
            output = out

            if step == 0:
                temp = list(output.size())
                temp.append(data.size()[-1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)

            outputsum[:, :, :, :, step] = output

        if self.onlyLast:
            return output
        else:
            return outputsum


class LIFPooling(nn.Module):
    def __init__(self,
                 kernel_size):
        super().__init__()

        self.avgPool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, data):

        for step in range(list(data.size())[-1]):
            out = data[:, :, :, :, step]
            out = self.avgPool(out)

            if step == 0:
                temp = list(out.size())
                temp.append(data.size()[-1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)
            outputsum[:, :, :, :, step] = out

        return outputsum


def test():
    thresh = 0.3
    lens = 0.5
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.ones(4, 2, 2, 2, 3).to(device)
    model = LIFConvGRU((2, 2), 2, 3, (3,3), spikeActFun=ActFun.apply).to(device)
    # model = LIFPooling(2).to(device)
    output = model(input)
    print(output.shape)


if __name__ == '__main__':
    test()
