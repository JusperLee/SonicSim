import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import abc
import re
import six

def expect_token_number(instr, token):
    first_token = re.match(r'^\s*' + token, instr)
    if first_token is None:
        return None
    instr = instr[first_token.end():]
    lr = re.match(r'^\s*(-?\d+\.?\d*e?-?\d*?)', instr)
    if lr is None:
        return None
    return instr[lr.end():], lr.groups()[0]


def expect_kaldi_matrix(instr):
    pos2 = instr.find('[', 0)
    pos3 = instr.find(']', pos2)
    mat = []
    for stt in instr[pos2 + 1:pos3].split('\n'):
        tmp_mat = np.fromstring(stt, dtype=np.float32, sep=' ')
        if tmp_mat.size > 0:
            mat.append(tmp_mat)
    return instr[pos3 + 1:], np.array(mat)



def to_kaldi_matrix(np_mat):
    """ function that transform as str numpy mat to standard kaldi str matrix

    Args:
        np_mat: numpy mat
    """
    np.set_printoptions(threshold=np.inf, linewidth=np.nan)
    out_str = str(np_mat)
    out_str = out_str.replace('[', '')
    out_str = out_str.replace(']', '')
    return '[ %s ]\n' % out_str


@six.add_metaclass(abc.ABCMeta)
class LayerBase(nn.Module):

    def __init__(self):
        super(LayerBase, self).__init__()

    @abc.abstractmethod
    def to_kaldi_nnet(self):
        pass


class UniDeepFsmn(LayerBase):

    def __init__(self, input_dim, output_dim, lorder=1, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(
            output_dim,
            output_dim, (lorder, 1), (1, 1),
            groups=output_dim,
            bias=False)

    def forward(self, input):
        """

        Args:
            input: torch with shape: batch (b) x sequence(T) x feature (h)

        Returns:
            batch (b) x channel (c) x sequence(T) x feature (h)
        """
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = torch.unsqueeze(p1, 1)
        # x: batch (b) x channel (c) x sequence(T) x feature (h)
        x_per = x.permute(0, 3, 2, 1)
        # x_per: batch (b) x feature (h) x sequence(T) x channel (c)
        y = F.pad(x_per, [0, 0, self.lorder - 1, 0])

        out = x_per + self.conv1(y)
        out1 = out.permute(0, 3, 2, 1)
        # out1: batch (b) x channel (c) x sequence(T) x feature (h)
        return input + out1.squeeze()

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<UniDeepFsmn> %d %d\n'\
                  % (self.output_dim, self.input_dim)
        re_str += '<LearnRateCoef> %d <HidSize> %d <LOrder> %d <LStride> %d <MaxNorm> 0\n'\
                  % (1, self.hidden_size, self.lorder, 1)

        lfiters = self.state_dict()['conv1.weight']
        x = np.flipud(lfiters.squeeze().numpy().T)
        re_str += to_kaldi_matrix(x)
        proj_weights = self.state_dict()['project.weight']
        x = proj_weights.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        linear_bias = self.state_dict()['linear.bias']
        x = linear_bias.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        return re_str

    def load_kaldi_nnet(self, instr):
        output = expect_token_number(
            instr,
            '<LearnRateCoef>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, lr = output

        output = expect_token_number(
            instr,
            '<HidSize>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, hiddensize = output
        self.hidden_size = int(hiddensize)

        output = expect_token_number(
            instr,
            '<LOrder>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, lorder = output
        self.lorder = int(lorder)

        output = expect_token_number(
            instr,
            '<LStride>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, lstride = output
        self.lstride = lstride

        output = expect_token_number(
            instr,
            '<MaxNorm>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error')

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('Fsmn format error')
        instr, mat = output
        mat1 = np.fliplr(mat.T).copy()
        self.conv1 = nn.Conv2d(
            self.output_dim,
            self.output_dim, (self.lorder, 1), (1, 1),
            groups=self.output_dim,
            bias=False)
        mat_th = torch.from_numpy(mat1).type(torch.FloatTensor)
        mat_th = mat_th.unsqueeze(1)
        mat_th = mat_th.unsqueeze(3)
        self.conv1.weight = torch.nn.Parameter(mat_th)

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, mat = output

        self.project = nn.Linear(self.hidden_size, self.output_dim, bias=False)
        self.linear = nn.Linear(self.input_dim, self.hidden_size)
        self.project.weight = torch.nn.Parameter(
            torch.from_numpy(mat).type(torch.FloatTensor))

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, mat = output
        self.linear.weight = torch.nn.Parameter(
            torch.from_numpy(mat).type(torch.FloatTensor))

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('UniDeepFsmn format error')
        instr, mat = output
        self.linear.bias = torch.nn.Parameter(
            torch.from_numpy(mat).type(torch.FloatTensor))
        return instr


class ComplexUniDeepFsmn(nn.Module):

    def __init__(self, nIn, nHidden=128, nOut=128):
        super(ComplexUniDeepFsmn, self).__init__()

        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_re_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)
        self.fsmn_im_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)

    def forward(self, x):
        r"""

        Args:
            x: torch with shape [batch, channel, feature, sequence, 2], eg: [6, 256, 1, 106, 2]

        Returns:
            [batch, feature, sequence, 2], eg: [6, 99, 1024, 2]
        """
        #
        b, c, h, T, d = x.size()
        x = torch.reshape(x, (b, c * h, T, d))
        # x: [b,h,T,2], [6, 256, 106, 2]
        x = torch.transpose(x, 1, 2)
        # x: [b,T,h,2], [6, 106, 256, 2]

        real_L1 = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary_L1 = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])
        # GRU output: [99, 6, 128]
        real = self.fsmn_re_L2(real_L1) - self.fsmn_im_L2(imaginary_L1)
        imaginary = self.fsmn_re_L2(imaginary_L1) + self.fsmn_im_L2(real_L1)
        # output: [b,T,h,2], [99, 6, 1024, 2]
        output = torch.stack((real, imaginary), dim=-1)

        # output: [b,h,T,2], [6, 99, 1024, 2]
        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (b, c, h, T, d))

        return output


class ComplexUniDeepFsmn_L1(nn.Module):

    def __init__(self, nIn, nHidden=128, nOut=128):
        super(ComplexUniDeepFsmn_L1, self).__init__()
        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)

    def forward(self, x):
        r"""

        Args:
            x: torch with shape [batch, channel, feature, sequence, 2], eg: [6, 256, 1, 106, 2]
        """
        b, c, h, T, d = x.size()
        # x : [b,T,h,c,2]
        x = torch.transpose(x, 1, 3)
        x = torch.reshape(x, (b * T, h, c, d))

        real = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])
        # output: [b*T,h,c,2], [6*106, h, 256, 2]
        output = torch.stack((real, imaginary), dim=-1)

        output = torch.reshape(output, (b, T, h, c, d))
        output = torch.transpose(output, 1, 3)
        return output


def get_casual_padding1d():
    pass

def get_casual_padding2d():
    pass

class cPReLU(nn.Module):

    def __init__(self, complex_axis=1):
        super(cPReLU,self).__init__()
        self.r_prelu = nn.PReLU()        
        self.i_prelu = nn.PReLU()
        self.complex_axis = complex_axis


    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2,self.complex_axis)
        real = self.r_prelu(real)
        imag = self.i_prelu(imag)
        return torch.cat([real,imag],self.complex_axis)

class NavieComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, projection_dim=None, bidirectional=False, batch_first=False):
        super(NavieComplexLSTM, self).__init__()

        self.input_dim = input_size//2
        self.rnn_units = hidden_size//2
        self.real_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional, batch_first=False)
        self.imag_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional, batch_first=False)
        if bidirectional:
            bidirectional=2
        else:
            bidirectional=1
        if projection_dim is not None:
            self.projection_dim = projection_dim//2 
            self.r_trans = nn.Linear(self.rnn_units*bidirectional, self.projection_dim)
            self.i_trans = nn.Linear(self.rnn_units*bidirectional, self.projection_dim)
        else:
            self.projection_dim = None

    def forward(self, inputs):
        if isinstance(inputs,list):
            real, imag = inputs 
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs,-1)
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out 
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        #print(real_out.shape,imag_out.shape)
        return [real_out, imag_out]
    
    def flatten_parameters(self):
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()

def complex_cat(inputs, axis):
    
    real, imag = [],[]
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data,2,axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real,axis)
    imag = torch.cat(imag,axis)
    outputs = torch.cat([real, imag],axis)
    return outputs

class ComplexConv2d(nn.Module):

    def __init__(
                    self,
                    in_channels,
                    out_channels,
                    kernel_size=(1,1),
                    stride=(1,1),
                    padding=(0,0),
                    dilation=1,
                    groups = 1,
                    causal=True, 
                    complex_axis=1,
                ):
        '''
            in_channels: real+imag
            out_channels: real+imag 
            kernel_size : input [B,C,D,T] kernel size in [D,T]
            padding : input [B,C,D,T] padding in [D,T]
            causal: if causal, will padding time dimension's left side,
                    otherwise both

        '''
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels//2
        self.out_channels = out_channels//2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis=complex_axis        
        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,padding=[self.padding[0],0],dilation=self.dilation, groups=self.groups)  
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,padding=[self.padding[0],0],dilation=self.dilation, groups=self.groups)  
        
        nn.init.normal_(self.real_conv.weight.data,std=0.05)
        nn.init.normal_(self.imag_conv.weight.data,std=0.05) 
        nn.init.constant_(self.real_conv.bias,0.) 
        nn.init.constant_(self.imag_conv.bias,0.) 
        

    def forward(self,inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs,[self.padding[1], 0,0,0]) 
        else:
            inputs = F.pad(inputs,[self.padding[1], self.padding[1],0,0]) 

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real,imag2real = torch.chunk(real,2, self.complex_axis)
            real2imag,imag2imag = torch.chunk(imag,2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real,imag = torch.chunk(inputs, 2, self.complex_axis)
        
            real2real = self.real_conv(real,)
            imag2imag = self.imag_conv(imag,)
        
            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)
        
        return out


class ComplexConvTranspose2d(nn.Module):

    def __init__(
                    self,
                    in_channels,
                    out_channels,
                    kernel_size=(1,1),
                    stride=(1,1),
                    padding=(0,0),
                    output_padding=(0,0),
                    causal=False,
                    complex_axis=1,
                    groups=1
                ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(ComplexConvTranspose2d, self).__init__()
        self.in_channels = in_channels//2
        self.out_channels = out_channels//2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding=output_padding 
        self.groups = groups 
        
        self.real_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels,kernel_size, self.stride,padding=self.padding,output_padding=output_padding, groups=self.groups)  
        self.imag_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels,kernel_size, self.stride,padding=self.padding,output_padding=output_padding, groups=self.groups)  
        self.complex_axis=complex_axis        

        nn.init.normal_(self.real_conv.weight,std=0.05)
        nn.init.normal_(self.imag_conv.weight,std=0.05) 
        nn.init.constant_(self.real_conv.bias,0.) 
        nn.init.constant_(self.imag_conv.bias,0.) 

    def forward(self,inputs):
         
        if isinstance(inputs, torch.Tensor):
            real,imag = torch.chunk(inputs, 2, self.complex_axis)
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            real = inputs[0]
            imag = inputs[1]
        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real,imag2real = torch.chunk(real,2, self.complex_axis)
            real2imag,imag2imag = torch.chunk(imag,2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real,imag = torch.chunk(inputs, 2, self.complex_axis)
        
            real2real = self.real_conv(real,)
            imag2imag = self.imag_conv(imag,)
        
            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)
        
        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)
        
        return out



# Source: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch 
# from https://github.com/IMLHF/SE_DCUNet/blob/f28bf1661121c8901ad38149ea827693f1830715/models/layers/complexnn.py#L55

class ComplexBatchNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
            track_running_stats=True, complex_axis=1):
        super(ComplexBatchNorm, self).__init__()
        self.num_features        = num_features//2
        self.eps                 = eps
        self.momentum            = momentum
        self.affine              = affine
        self.track_running_stats = track_running_stats 
        
        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br  = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi  = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br',  None)
            self.register_parameter('Bi',  None)
        
        if self.track_running_stats:
            self.register_buffer('RMr',  torch.zeros(self.num_features))
            self.register_buffer('RMi',  torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones (self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones (self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr',                 None)
            self.register_parameter('RMi',                 None)
            self.register_parameter('RVrr',                None)
            self.register_parameter('RVri',                None)
            self.register_parameter('RVii',                None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9) # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert(xr.shape == xi.shape)
        assert(xr.size(1) == self.num_features)

    def forward(self, inputs):
        #self._check_input_dim(xr, xi)
        
        xr, xi = torch.chunk(inputs,2, axis=self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i!=1]
        vdim  = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr-Mr, xi-Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr   = Vrr + self.eps
        Vri   = Vri
        Vii   = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau   = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s     = delta.sqrt()
        t     = (tau + 2*s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst   = (s * t).reciprocal()
        Urr   = (s + Vii) * rst
        Uii   = (s + Vrr) * rst
        Uri   = (  - Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__) 

def complex_cat(inputs, axis):
    
    real, imag = [],[]
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data,2,axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real,axis)
    imag = torch.cat(imag,axis)
    outputs = torch.cat([real, imag],axis)
    return outputs

        
class ComplexBatchNorm2d(nn.Module):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(
            num_features=num_features,
            momentum=momentum,
            affine=affine,
            eps=eps,
            track_running_stats=track_running_stats,
            **kwargs)
        self.bn_im = nn.BatchNorm2d(
            num_features=num_features,
            momentum=momentum,
            affine=affine,
            eps=eps,
            track_running_stats=track_running_stats,
            **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output
