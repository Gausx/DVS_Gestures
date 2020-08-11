import torch
from torch import nn
from module import convLstm, convRnns, convGru
from torch.autograd import Variable

scale = 0.0  # 膜电位初始化的噪音，0是0初始化


torch.manual_seed(666)  # 随机种子
torch.cuda.manual_seed_all(666)


# 相当于设置遗忘率，用于spike替代sigmoid
# 这个参数在现有的LSTM模型中没有被采用
#######################################################
# init method
def paramInit(model, method='xavier'):
    scale = 0.05
    for name, w in model.named_parameters():
        if 'weight' in name:
            if method == 'xavier':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)
                w *= scale
        elif 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            pass
    # scale = 0.05 #权重初始化的情况
    # wsize = weight.size()
    # bsize = bias.size()
    # w = torch.randn(weight.size(),device=weight.device)*scale
    # b = torch.randn(bias.size(),device=bias.device)*scale
    # return w,b
#######################################################


class LIFRNNCell(nn.Module):
    def __init__(
            self,
            inputSize,
            hiddenSize,
            spikeActFun,
            dropOut=0,
            useBatchNorm=False,
            useLayerNorm=False,
            decay=0.3):
        """
        :param inputSize:(Num) number of input
        :param hiddenSize:(Num) number of output
        :param actFun:handle of activation function
        :param spikeAcFun:handle of recurrent spike firing function
        :param dropOut: 0~1 unused
        :param useBatchNorm:
        :param useLayerNorm:
        :param init_method:
        :param gFun:
        """
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.decay = decay
        self.spikeActFun = spikeActFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.UseDropOut = True
        self.batchSize = None
        # block 1. add synaptic inputs:Wx+b
        self.rnncell = nn.RNNCell(input_size=inputSize, hidden_size=hiddenSize)
        # block 3. add a Norm layer
        if self.useBatchNorm:
            self.BNLayerx = nn.BatchNorm1d(hiddenSize)
            self.BNLayerc = nn.BatchNorm1d(hiddenSize)
        if self.useLayerNorm:
            self.Lnormx = nn.LayerNorm(hiddenSize)
            self.Lnormc = nn.LayerNorm(hiddenSize)
        # block 4. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

        self.h = None

    def forward(self, input, init_v=None):
        """
        :param input:
        :param init_v:
        :return:
        """
        self.batchSize = input.size()[0]
        input = input.reshape([self.batchSize, -1])
        torch.cuda.empty_cache()

        if input.device != self.rnncell.weight_hh.device:
            input = input.to(self.rnncell.weight_hh.device)

        if self.h is None:
            if init_v is None:
                self.h = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     device=input.device)
            else:
                self.h = init_v.clone()

        # Step 1: accumulate and reset,spike used as forgetting gate
        u = self.rnncell(input, self.h)
        # Step 2: renew
        self.h = self.decay * u * (1 - self.spikeActFun(u))
        x = self.spikeActFun(u)

        # step 3: Norms
        if self.useBatchNorm:
            self.h = self.BNLayerc(self.h)
            x = self.BNLayerx(x)
        if self.useLayerNorm:
            self.h = self.Lnormc(self.h)
            x = self.Lnormx(x)
        # step 4:dropout
        if self.UseDropOut:
            x = self.DPLayer(x)
        return x

    def reset(self):
        self.h = None


class LIFLSTMCell(nn.Module):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 spikeActFun,
                 dropOut=0,
                 useBatchNorm=False,
                 useLayerNorm=False,
                 decay=0.3):
        """
        :param inputSize:(Num) number of input
        :param hiddenSize:(Num) number of output
        :param spikeAcFun:handle of recurrent spike firing function
        :param dropOut: 0~1 unused
        :param useBatchNorm:
        :param useLayerNorm:
        """
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.decay = decay
        self.spikeActFun = spikeActFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.UseDropOut = True
        self.batchSize = None

        self.lstmcell = nn.LSTMCell(inputSize, hiddenSize)

        # block 3. add a Norm layer
        if self.useBatchNorm:
            self.BNLayerx = nn.BatchNorm1d(hiddenSize)
            self.BNLayerc = nn.BatchNorm1d(hiddenSize)
        if self.useLayerNorm:
            self.Lnormx = nn.LayerNorm(hiddenSize)
            self.Lnormc = nn.LayerNorm(hiddenSize)

        # block 4. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

        self.c = None
        self.h = None

    def forward(self, input, init_v=None):
        """
        :param input:
        :param init_v:
        :return:
        """
        self.batchSize = input.size()[0]

        input = input.reshape([self.batchSize, -1])

        torch.cuda.empty_cache()

        if input.device != self.lstmcell.weight_hh.device:
            input = input.to(self.lstmcell.weight_hh.device)

        # 初始化隐藏状态
        if self.h is None:
            if init_v is None:
                self.c = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     device=input.device)
                self.h = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     device=input.device)

            else:
                self.c = init_v.clone()
                self.h = init_v.clone()

        # Step 1: accumulate and reset,spike used as forgetting gate
        u, self.c = self.lstmcell(input, (self.h, self.c))

        # Step 2: renew
        x = self.spikeActFun(u)
        self.h = self.decay * u * (1 - self.spikeActFun(u))    # 重要


        # step 3: Norms    后续看能否使用
        if self.useBatchNorm:
            self.c = self.BNLayerc(self.c)
            self.h = self.BNLayerc(self.h)
            x = self.BNLayerx(x)
        if self.useLayerNorm:
            self.c = self.Lnormc(self.c)
            self.h = self.Lnormc(self.h)
            x = self.Lnormx(x)

        # step 4:dropout
        if self.UseDropOut:
            x = self.DPLayer(x)
        return x

    def reset(self):
        self.c = None
        self.h = None


class LIFGRUCell(nn.Module):
    def __init__(
            self,
            inputSize,
            hiddenSize,
            spikeActFun,
            dropOut=0,
            useBatchNorm=False,
            useLayerNorm=False,
            decay=0.3):
        """
        :param inputSize:(Num) number of input
        :param hiddenSize:(Num) number of output
        :param actFun:handle of activation function
        :param spikeAcFun:handle of recurrent spike firing function
        :param dropOut: 0~1 unused
        :param useBatchNorm:
        :param useLayerNorm:
        :param init_method:
        :param gFun:
        """
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.decay = decay
        self.spikeActFun = spikeActFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.UseDropOut = True
        self.batchSize = None
        # block 1. add synaptic inputs:Wx+b
        self.grucell = nn.GRUCell(input_size=inputSize, hidden_size=hiddenSize)

        # block 3. add a Norm layer
        if self.useBatchNorm:
            self.BNLayerx = nn.BatchNorm1d(hiddenSize)
            self.BNLayerc = nn.BatchNorm1d(hiddenSize)
        if self.useLayerNorm:
            self.Lnormx = nn.LayerNorm(hiddenSize)
            self.Lnormc = nn.LayerNorm(hiddenSize)
        # block 4. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

        self.h = None

    def forward(self, input, init_v=None):
        """
        :param input:
        :param init_v:
        :return:
        """
        self.batchSize = input.size()[0]
        input = input.reshape([self.batchSize, -1])
        torch.cuda.empty_cache()
        if input.device != self.grucell.weight_hh.device:
            input = input.to(self.grucell.weight_hh.device)
        if self.h is None:
            if init_v is None:
                self.h = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     device=input.device)
            else:
                self.h = init_v.clone()
        # Step 1: accumulate and reset,spike used as forgetting gate
        u = self.grucell(input, self.h)

        # Step 2: renew
        x = self.spikeActFun(u)
        self.h = self.decay * u * (1 - self.spikeActFun(u))
        # step 3: Norms
        if self.useBatchNorm:
            self.h = self.BNLayerc(self.h)
            x = self.BNLayerx(x)
        if self.useLayerNorm:
            self.h = self.Lnormc(self.h)
            x = self.Lnormx(x)

        # step 4:dropout
        if self.UseDropOut:
            x = self.DPLayer(x)
        return x

    def reset(self):
        self.h = None


class LIFConvRNNCell(nn.Module):
    def __init__(self,
                 input_range,
                 inputSize,
                 hiddenSize,
                 kernel_size,
                 spikeActFun,
                 dropOut=0,
                 bias=True,
                 useBatchNorm=False,
                 useLayerNorm=False,
                 decay=0.3):
        """
        :param inputSize:(Num) number of input
        :param hiddenSize:(Num) number of output
        :param spikeAcFun:handle of recurrent spike firing function
        :param dropOut: 0~1 unused
        :param useBatchNorm:
        :param useLayerNorm:
        """
        super().__init__()
        self.height, self.width = input_range
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.decay = decay
        self.spikeActFun = spikeActFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.UseDropOut = True
        self.batchSize = None

        self._check_kernel_size_consistency(kernel_size)

        self.convrnnscell = convRnns.ConvRNNCell(input_size=input_range,
                                                 input_dim=inputSize,
                                                 hidden_dim=hiddenSize,
                                                 kernel_size=kernel_size,
                                                 bias=bias)

        # block 3. add a Norm layer
        if self.useBatchNorm:
            self.BNLayerx = nn.BatchNorm1d(hiddenSize)
            self.BNLayerc = nn.BatchNorm1d(hiddenSize)
        if self.useLayerNorm:
            self.Lnormx = nn.LayerNorm(hiddenSize)
            self.Lnormc = nn.LayerNorm(hiddenSize)

        # block 4. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

        self.h = None

    def forward(self, input, init_v=None):
        """
        :param input:
        :param init_v:
        :return:
        """
        self.batchSize = input.size()[0]

        torch.cuda.empty_cache()

        if input.device != self.convrnnscell.conv.weight.device:
            input = input.to(self.convrnnscell.conv.weight.device)

        # 初始化隐藏状态
        if self.h is None:
            if init_v is None:
                self.h = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     self.height,
                                     self.width,
                                     device=input.device)

            else:
                self.h = init_v.clone()

        # Step 1: accumulate and reset,spike used as forgetting gate
        u = self.convrnnscell(input_tensor=input,
                              cur_state=[self.h])

        # Step 2: renew
        x = self.spikeActFun(u)
        self.h = self.decay * u * (1 - self.spikeActFun(u))    # 重要

        # step 3: Norms    后续看能否使用
        if self.useBatchNorm:
            self.h = self.BNLayerc(self.h)
            x = self.BNLayerx(x)
        if self.useLayerNorm:
            self.h = self.Lnormc(self.h)
            x = self.Lnormx(x)

        # step 4:dropout
        if self.UseDropOut:
            x = self.DPLayer(x)

        return x

    def reset(self):
        self.h = None

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all(
                [isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')


class LIFConvLSTMCell(nn.Module):
    def __init__(self,
                 input_range,
                 inputSize,
                 hiddenSize,
                 kernel_size,
                 spikeActFun,
                 dropOut=0,
                 bias=True,
                 useBatchNorm=False,
                 useLayerNorm=False,
                 decay=0.3):
        """
        :param inputSize:(Num) number of input
        :param hiddenSize:(Num) number of output
        :param spikeAcFun:handle of recurrent spike firing function
        :param dropOut: 0~1 unused
        :param useBatchNorm:
        :param useLayerNorm:
        """
        super().__init__()
        self.height, self.width = input_range
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.decay = decay
        self.spikeActFun = spikeActFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.UseDropOut = True
        self.batchSize = None

        self._check_kernel_size_consistency(kernel_size)

        self.convlstmcell = convLstm.ConvLSTMCell(input_size=input_range,
                                                  input_dim=inputSize,
                                                  hidden_dim=hiddenSize,
                                                  kernel_size=kernel_size,
                                                  bias=bias)

        # block 3. add a Norm layer
        if self.useBatchNorm:
            self.BNLayerx = nn.BatchNorm1d(hiddenSize)
            self.BNLayerc = nn.BatchNorm1d(hiddenSize)

        if self.useLayerNorm:
            self.Lnormx = nn.LayerNorm(hiddenSize)
            self.Lnormc = nn.LayerNorm(hiddenSize)

        # block 4. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

        self.c = None
        self.h = None

    def forward(self, input, init_v=None):
        """
        :param input:
        :param init_v:
        :return:
        """
        self.batchSize = input.size()[0]

        torch.cuda.empty_cache()

        if input.device != self.convlstmcell.conv.weight.device:
            input = input.to(self.convlstmcell.conv.weight.device)

        # 初始化隐藏状态
        if self.h is None:
            if init_v is None:
                self.h = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     self.height,
                                     self.width,
                                     device=input.device)

                self.c = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     self.height,
                                     self.width,
                                     device=input.device)

            else:
                self.h, self.c = init_v.clone()

        # Step 1: accumulate and reset,spike used as forgetting gate
        u, self.c = self.convlstmcell(input_tensor=input,
                                      cur_state=[self.h, self.c])

        # Step 2: renew
        # x = self.spikeActFun(u)
        self.h = self.decay * u * (1 - self.spikeActFun(u))    # 重要
        # # TODO: 测试C进行衰减
        # self.c = self.decay * self.c * (1- self.spikeActFun(self.c))
        # self.c = self.decay * self.c
        x = u
        # self.h = u

        # step 3: Norms    后续看能否使用
        if self.useBatchNorm:
            self.c = self.BNLayerc(self.c)
            self.h = self.BNLayerc(self.h)
            x = self.BNLayerx(x)
        if self.useLayerNorm:
            self.c = self.Lnormc(self.c)
            self.h = self.Lnormc(self.h)
            x = self.Lnormx(x)

        # step 4:dropout
        if self.UseDropOut:
            x = self.DPLayer(x)
        return x

    def reset(self):
        self.c = None
        self.h = None

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all(
                [isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')


class LIFConvGRUCell(nn.Module):
    def __init__(self,
                 input_range,
                 inputSize,
                 hiddenSize,
                 kernel_size,
                 spikeActFun,
                 dropOut=0,
                 bias=True,
                 useBatchNorm=False,
                 useLayerNorm=False,
                 decay=0.3):
        """
        :param inputSize:(Num) number of input
        :param hiddenSize:(Num) number of output
        :param actFun:handle of activation function
        :param spikeAcFun:handle of recurrent spike firing function
        :param dropOut: 0~1 unused
        :param useBatchNorm:
        :param useLayerNorm:
        :param init_method:
        :param gFun:
        """
        super().__init__()
        self.height, self.width = input_range
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.decay = decay
        self.spikeActFun = spikeActFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.UseDropOut = True
        self.batchSize = None

        self._check_kernel_size_consistency(kernel_size)

        # block 1. add synaptic inputs:Wx+b
        self.convgrucell = convGru.ConvGRUCell(input_size=input_range,
                                               input_dim=inputSize,
                                               hidden_dim=hiddenSize,
                                               kernel_size=kernel_size,
                                               bias=bias)

        # block 3. add a Norm layer
        if self.useBatchNorm:
            self.BNLayerx = nn.BatchNorm1d(hiddenSize)
            self.BNLayerc = nn.BatchNorm1d(hiddenSize)
        if self.useLayerNorm:
            self.Lnormx = nn.LayerNorm(hiddenSize)
            self.Lnormc = nn.LayerNorm(hiddenSize)
        # block 4. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

        self.h = None

    def forward(self, input, init_v=None):
        """
        :param input:
        :param init_v:
        :return:
        """
        self.batchSize = input.size()[0]

        torch.cuda.empty_cache()

        if input.device != self.convgrucell.conv1.weight.device:
            input = input.to(self.convgrucell.conv1.weight.device)

        if self.h is None:
            if init_v is None:
                self.h = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     self.height,
                                     self.width,
                                     device=input.device)
            else:
                self.h = init_v.clone()

        # Step 1: accumulate and reset,spike used as forgetting gate
        u = self.convgrucell(input_tensor=input,
                             cur_state=[self.h])

        # Step 2: renew
        x = self.spikeActFun(u)
        self.h = self.decay * u * (1 - self.spikeActFun(u))

        # step 3: Norms
        if self.useBatchNorm:
            self.h = self.BNLayerc(self.h)
            x = self.BNLayerx(x)
        if self.useLayerNorm:
            self.h = self.Lnormc(self.h)
            x = self.Lnormx(x)

        # step 4:dropout
        if self.UseDropOut:
            x = self.DPLayer(x)
        return x

    def reset(self):
        self.h = None

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all(
                [isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
