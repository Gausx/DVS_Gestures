'''

'''

import torch.nn as nn
from torch.autograd import Variable
import torch


class ConvGRUCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvGRU cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvGRUCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv1 = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                               out_channels=2 * self.hidden_dim,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               bias=self.bias)
        self.conv2 = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                               out_channels=self.hidden_dim,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               bias=self.bias)

    def forward(self, input_tensor, cur_state):

        # concatenate along channel axis
        combined = torch.cat([input_tensor, cur_state[0]], dim=1)

        combined_conv = self.conv1(combined)
        z, r = torch.split(combined_conv,
                           self.hidden_dim,
                           dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        combined2 = torch.cat([input_tensor, r * cur_state[0]], dim=1)
        h_ = torch.tanh(self.conv2(combined2))

        h_next = (1 - z) * cur_state[0] + z * h_

        torch.cuda.empty_cache()

        return h_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size,
                                     self.hidden_dim,
                                     self.height,
                                     self.width)).cuda())


class ConvGRU(nn.Module):

    def __init__(
            self,
            input_size,
            input_dim,
            hidden_dim,
            kernel_size,
            num_layers,
            batch_first=False,
            bias=True,
            return_all_layers=False):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having
        # len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            # 用来创造一系列的卷积层，其中上一个层的隐层维度是下一个层的输入维度
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:  # batch first参数：batch的大小是第一个维度
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        # 为了实现结构化的设计，cell需要和batch无关，因此要交给LSTM模型顶层来初始化隐层来符合batchSize
        if hidden_state is not None:
            raise NotImplementedError()  # 为了实现训练时对状态进行修改？
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)  # 第二维 时间序列长度
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output  # 把每一个时间步的输出状态叠加成下一层的输入

            layer_output_list.append(layer_output)  # 每一个时间步的状态表
            last_state_list.append([h])  # 最终状态列表

        if not self.return_all_layers:  # 如果不需要返回所有层的话 就只返回最后一个就好
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all(
                [isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
