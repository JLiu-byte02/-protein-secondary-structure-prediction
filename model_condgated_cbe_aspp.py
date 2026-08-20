from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import Conv1d
from torch.nn.modules.conv import _ConvNd
from torch import nn
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
# from config import config


class GatedCBEASPP(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 kernel_width,
                 out_channel,
                 n_layers,
                 res_block_cnt,
                 activationFunc,
                 config,
                 dropout=0.1):
        super(GatedCBEASPP, self).__init__()

        aspp_layer_num = len(config.dilation_list)
        if activationFunc == 'logsigmoid':
            actFunc = nn.LogSigmoid()
        elif activationFunc == 'sigmoid':
            actFunc = nn.Sigmoid()
        elif activationFunc == 'tanh':
            actFunc = nn.Tanh()
        elif activationFunc == 'relu':
            actFunc = nn.ReLU()
        elif activationFunc == 'none':
            actFunc = None

        self.res_block_cnt = res_block_cnt
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.padding_left = nn.ConstantPad1d((kernel_width - 1, 0), 0)

        self.conv_0 = nn.Sequential(CondConv1D(in_channels=embed_dim, out_channels=out_channel,
                                               kernel_size=kernel_width, stride=1,
                                               padding=0, dilation=1, num_experts=3, dropout_rate=config.cond_dropout_rate)
                                    # , actFunc
                                    )
        self.b_0 = nn.Parameter(torch.zeros(out_channel, 1))  # same as paper
        self.conv_gate_0 = nn.Sequential(
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=out_channel,
                kernel_size=kernel_width)
            #  , actFunc
        )
        self.c_0 = nn.Parameter(torch.zeros(out_channel, 1))

        self.convs = nn.ModuleList([nn.Sequential(CondConv1D(in_channels=out_channel, out_channels=out_channel,
                                                             kernel_size=kernel_width, stride=1,
                                                             padding=0, dilation=1, num_experts=3, dropout_rate=config.cond_dropout_rate)
                                                  #   , actFunc

                                                  ) for _ in range(n_layers)
                                    ])

        self.bs = nn.ParameterList([
            nn.Parameter(torch.zeros(out_channel, 1))  # collections of b
            for _ in range(n_layers)
        ])

        self.conv_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_width)
                #    , actFunc
            ) for _ in range(n_layers)
        ])

        self.cs = nn.ParameterList([
            nn.Parameter(torch.zeros(out_channel, 1)) for _ in range(n_layers)
        ])

        self.fc = nn.Linear(out_channel, vocab_size)
        self.dropout = nn.Dropout(p=dropout)  # todo use dropout

        # conv1d Input: (N, Cin, Lin)
        # constantpad1d Input: (N,C,Win)  Output: (N,C,Wout)

        self.conv_top = nn.Sequential(
            nn.Conv1d(
                in_channels=(out_channel + config.lstm_hidden_size * 2) *
                (aspp_layer_num + 1),
                out_channels=vocab_size,
                kernel_size=config.top_window_size,
                padding=int((config.top_window_size - 1) / 2)),
            #   nn.BatchNorm1d(num_features=config.class_num),
            nn.Dropout(config.output_dropout_rate),
            nn.LogSigmoid()
            # nn.Sigmoid()
        )

        self.conv_end = config.conv_end
        self.resnet = config.resnet

        # BiLSTM decoder configuration
        self.bilstm_hidden_size = getattr(
            config, 'transformer_hidden_size', getattr(config, 'lstm_hidden_size', 512))
        self.bilstm_num_layers = getattr(
            config, 'bilstm_layer_num', getattr(config, 'blstm_layer_num', 4))
        self.bilstm_dropout = getattr(
            config, 'bilstm_dropout', getattr(config, 'lstm_dropout_rate', 0.1))

        # Input projection: feature_size -> hidden_size
        self.input_projection = nn.Linear(config.feature_size, self.bilstm_hidden_size)

        # Manually stacked BiLSTM layers (each with num_layers=1, bidirectional=True)
        self.bilstm_layers = nn.ModuleList()
        for i in range(self.bilstm_num_layers):
            input_size = self.bilstm_hidden_size if i == 0 else self.bilstm_hidden_size * 2
            self.bilstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=self.bilstm_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                    bidirectional=True,
                )
            )

        # LayerNorm after each layer
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.bilstm_hidden_size * 2) for _ in range(self.bilstm_num_layers)]
        )

        # Multi-head self-attention (to enhance long-term dependencies)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.bilstm_hidden_size * 2,
            num_heads=8,
            dropout=self.bilstm_dropout,
            batch_first=True,
        )

        # Output projection layer
        self.output_projection = nn.Linear(self.bilstm_hidden_size * 2, self.bilstm_hidden_size * 2)


        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        for lstm in self.bilstm_layers:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)

        # self.fc1 = nn.Sequential(
        #     nn.Sigmoid(),
        #     nn.Dropout(config.fc1_dropout_rate),
        #     # nn.Linear(config.feature_size*2, config.class_num),
        #     nn.Linear(config.node_size + config.lstm_hidden_size * 2,
        #               config.class_num),
        #     nn.ReLU())
        # # self.fc2 = nn.Sequential(
        # #     nn.Linear(config.class_num, config.class_num)
        # #     # ,nn.LogSoftmax(dim=config.class_num)
        # # )

        self.convs_aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=out_channel + config.lstm_hidden_size * 2,
                    out_channels=out_channel + config.lstm_hidden_size * 2,
                    kernel_size=aspp_window_size,
                    padding=int((aspp_window_size - 1) / 2 +
                                (dil - 1) * int((aspp_window_size - 1) / 2)),
                    # padding=int((config.bot_window_size-1)/2),
                    dilation=dil),
                nn.ReLU(),
                # af,
                nn.BatchNorm1d(out_channel + config.lstm_hidden_size * 2),

                # nn.Sigmoid()
            ) for dil, aspp_window_size in zip(config.dilation_list,
                                               config.window_size_list)
        ])

        self.conv_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=(out_channel + config.lstm_hidden_size * 2) *
                (aspp_layer_num + 1),
                out_channels=config.class_num,
                kernel_size=1),
            #   nn.BatchNorm1d(num_features=config.node_size),
            nn.LogSigmoid(),
            nn.Dropout(config.output_dropout_rate),
            #   nn.BatchNorm1d(num_features=config.class_num) # after changed
            # nn.Sigmoid()
        )

        self.device = config.device
        # self.hidden_size = config.feature_size
        self.lstm_hidden_size = config.lstm_hidden_size 
        self.feature_size = config.feature_size 
        self.batch_size = config.batch_size 
        self.data_max_text_len = config.data_max_text_len 
        self.blstm_layer_num = config.blstm_layer_num 
        self.output_layer = config.output_layer 
        # BiGRU Encoder configuration (replacing CondGCNN Block × M)
        self.bigru_layer_num = getattr(config, 'bigru_layer_num', 3)
        self.bigru_encoder = nn.GRU(
            input_size=config.feature_size,
            hidden_size=out_channel,
            num_layers=self.bigru_layer_num,
            batch_first=True,
            dropout=config.lstm_dropout_rate,
            bidirectional=True,
        )
        # Project bidirectional output (2*out_channel) back to out_channel
        self.bigru_proj = nn.Linear(out_channel * 2, out_channel)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.blstm_layer_num * 2, batch_size,
                            self.lstm_hidden_size).to(self.device),
                torch.zeros(self.blstm_layer_num * 2, batch_size,
                            self.lstm_hidden_size).to(self.device))

    def forward(self, seq):
        # seq:(batch,seq_len)
        batch_size = seq.size(0)
        seq_len = seq.size(1)
        # x = self.embedding(seq)  # x: (batch,seq_len,embed_dim)
        # x.transpose_(1, 2)  # x:(batch,embed_dim,seq_len) , embed_dim equals to in_channel
        # Replace CondGCNN Block × M with BiGRU Encoder
        x = seq  # [batch, feature_size, seq_len]
        # BiGRU expects [batch, seq_len, feature_size]
        bigru_in = x.permute(0, 2, 1)
        bigru_out, _ = self.bigru_encoder(bigru_in)  # [batch, seq_len, 2*out_channel]
        bigru_out = self.bigru_proj(bigru_out)       # [batch, seq_len, out_channel]
        h = bigru_out.permute(0, 2, 1)               # [batch, out_channel, seq_len]

        x = seq
        x = x.permute(0, 2, 1)  # [batch, seq_len, feature_size]
        
        bilstm_in = self.input_projection(x)  # [batch, seq_len, hidden]
        h_l = bilstm_in
        for i, lstm in enumerate(self.bilstm_layers):
            out, (hidden_state, cell_state) = lstm(h_l)
            out = self.layer_norms[i](out)
            out = F.dropout(out, p=self.bilstm_dropout, training=self.training)
            h_l = out
        attended_h, _ = self.attention(h_l, h_l, h_l)
        bilstm_out = self.output_projection(attended_h + h_l)  # [batch, seq_len, 2*hidden]
        bilstm_out = bilstm_out.permute(0, 2, 1)
        backbone_out = torch.cat((h, bilstm_out), 1)

        # [32,X,700]
        for index, conv in enumerate(self.convs_aspp):
            out_each = conv(backbone_out)
            if index == 0:
                out = out_each
            else:
                out = torch.cat((out, out_each), 1)
        out = torch.cat((out, backbone_out), 1)

        if self.output_layer == 'conv1':
            logic = self.conv_1(out)
        elif self.output_layer == 'convtop':
            logic = self.conv_top(out)

        # if self.conv_end:
        #     logic = self.conv_top(h)
        # else:
        #     h.transpose_(1, 2)  # h:(batch,seq_len,out_channel)

        #     logic = self.fc(h)  # logic:(batch,seq_len,vocab_size)
        #     logic.transpose_(
        #         1, 2
        #     )  # logic:(batch,vocab_size,seq_len) cross_entropy input:(N,C,d1,d2,..) C is num of class
        return logic


class CondConv1D(_ConvNd):
    # class CondConv1D(Conv1d):
    r"""Learn specialized convolutional kernels for each example.
    As described in the paper
    `CondConv: Conditionally Parameterized Convolutions for Efficient Inference`_ ,
    conditionally parameterized convolutions (CondConv), 
    which challenge the paradigm of static convolutional kernels 
    by computing convolutional kernels as a function of the input.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts per layer 
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    .. _CondConv: Conditionally Parameterized Convolutions for Efficient Inference:
       https://arxiv.org/abs/1904.04971
    """

    def __init__(self, in_channels=42, out_channels=9, kernel_size=11, stride=1,
                 padding=int((11-1)/2), dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        # kernel_size = _pair(kernel_size)
        # stride = _pair(stride)
        # padding = _pair(padding)
        # dilation = _pair(dilation)

        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

        # super(CondConv1D, self).__init__(
        #     in_channels, out_channels, kernel_size, stride, padding, dilation,
        #     False, _pair(0), groups, bias, padding_mode)
        super(CondConv1D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(
            F.adaptive_avg_pool1d, output_size=(1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv1d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        # print('input', input.size())
        pooled_inputs = self._avg_pooling(input)
        # print('pooled_inputs', pooled_inputs.size())
        routing_weights = self._routing_fn(pooled_inputs)
        # print('routing_weights', routing_weights.size())
        # print('self.weight', self.weight.size())

        for idx, weight_sub in enumerate(self.weight):
            kernel_sub = weight_sub
            # kernel_sub = torch.squeeze(weight_sub, 0)
            output_0 = self._conv_forward(input, kernel_sub)
            if idx == 0:
                sumout = routing_weights[:, idx, None, None] * output_0
            else:
                # sumout = torch.sum(routing_weights[:, idx, None, None] * output_0 ,1)
                sumout += routing_weights[:, idx, None, None] * output_0
        # print('sumout', sumout.size())
        return sumout

        # first:([100, 42, 11])
        kernels = torch.sum(
            routing_weights[:, None, None, None] * self.weight, 0)
        # print('------kernels', kernels.size())
        # kernels = torch.sum(routing_weights[:,None, None, None, None] * self.weight, 0)

        return self._conv_forward(input, kernels)


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        # x = torch.flatten(x)
        x = torch.squeeze(x, 2)
        # print('flattenX', x.size())
        x = self.dropout(x)
        x = self.fc(x)

        return F.sigmoid(x)
