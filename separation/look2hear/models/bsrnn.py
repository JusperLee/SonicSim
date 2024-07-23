import torch
import torch.nn as nn
import numpy as np
from .base_model import BaseModel

class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(ResRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps
        
        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size*(int(bidirectional)+1), input_size)

    def forward(self, input):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(self.norm(input).transpose(1,2).contiguous())
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(input.shape[0], input.shape[2], input.shape[1])
        
        return input + rnn_output.transpose(1,2).contiguous()

class BSNet(nn.Module):
    def __init__(self, in_channel, nband=7):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband

        self.band_rnn = ResRNN(self.feature_dim, self.feature_dim*2)
        self.band_comm = ResRNN(self.feature_dim, self.feature_dim*2)

    def forward(self, input):
        # input shape: B, nband*N, T
        B, N, T = input.shape

        band_output = self.band_rnn(input.view(B*self.nband, self.feature_dim, -1)).view(B, self.nband, -1, T)

        # band comm
        band_output = band_output.permute(0,3,2,1).contiguous().view(B*T, -1, self.nband)
        output = self.band_comm(band_output).view(B, T, -1, self.nband).permute(0,3,2,1).contiguous()

        return output.view(B, N, T)

class BSRNN(BaseModel):
    def __init__(self, sample_rate=44100, win=2048, stride=512, feature_dim=128, num_repeat=12, num_output=4):
        super(BSRNN, self).__init__(sample_rate=sample_rate)
        
        self.sample_rate = sample_rate
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.num_output = num_output
        self.eps = torch.finfo(torch.float32).eps

        # 0-1k (50 hop), 1k-2k (100 hop), 2k-4k (250 hop), 4k-8k (500 hop)
        bandwidth_50 = int(np.floor(50 / (sample_rate / 2.) * self.enc_dim))
        bandwidth_100 = int(np.floor(100 / (sample_rate / 2.) * self.enc_dim))
        bandwidth_250 = int(np.floor(250 / (sample_rate / 2.) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sample_rate / 2.) * self.enc_dim))
        self.band_width = [bandwidth_50]*20
        self.band_width += [bandwidth_100]*10
        self.band_width += [bandwidth_250]*8
        self.band_width += [bandwidth_500]*8
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        print(self.band_width)
        
        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(nn.Sequential(nn.GroupNorm(1, self.band_width[i]*2, self.eps),
                                         nn.Conv1d(self.band_width[i]*2, self.feature_dim, 1)
                                        )
                          )

        self.separator = []
        for i in range(num_repeat):
            self.separator.append(BSNet(self.nband*self.feature_dim, self.nband))             
        self.separator = nn.Sequential(*self.separator)
        
        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(nn.Sequential(nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                                           nn.Conv1d(self.feature_dim, self.feature_dim*self.num_output, 1),
                                           nn.Tanh(),
                                           nn.Conv1d(self.feature_dim*self.num_output, self.feature_dim*2*self.num_output, 1, groups=self.num_output),
                                           nn.Tanh(),
                                           nn.Conv1d(self.feature_dim*2*self.num_output, self.band_width[i]*4*self.num_output, 1, groups=self.num_output)
                                          )
                            )

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest
        
    def forward(self, input):
        # input shape: (B, C, T)
        was_one_d = False
        if input.ndim == 1:
            was_one_d = True
            input = input.unsqueeze(0).unsqueeze(1)
        if input.ndim == 2:
            was_one_d = True
            input = input.unsqueeze(1)
        if input.ndim == 3:
            input = input
        batch_size, nch, nsample = input.shape
        input = input.view(batch_size*nch, -1)

        # frequency-domain separation
        spec = torch.stft(input, n_fft=self.win, hop_length=self.stride, 
                          window=torch.hann_window(self.win).to(input.device).type(input.type()),
                          return_complex=True)

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec_RI = []
        subband_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec_RI.append(spec_RI[:,:,band_idx:band_idx+self.band_width[i]].contiguous())
            subband_spec.append(spec[:,band_idx:band_idx+self.band_width[i]])  # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(self.BN[i](subband_spec_RI[i].view(batch_size*nch, self.band_width[i]*2, -1)))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T
        # import pdb; pdb.set_trace()
        # separator
        sep_output = self.separator(subband_feature.view(batch_size*nch, self.nband*self.feature_dim, -1))  # B, nband*N, T
        sep_output = sep_output.view(batch_size*nch, self.nband, self.feature_dim, -1)
        
        sep_subband_spec = []
        for i in range(self.nband):
            this_output = self.mask[i](sep_output[:,i]).view(batch_size*nch, 2, 2, self.num_output, self.band_width[i], -1)
            this_mask = this_output[:,0] * torch.sigmoid(this_output[:,1])  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:,0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:,1]  # B*nch, K, BW, T
            # force mask sum to 1
            this_mask_real_sum = this_mask_real.sum(1).unsqueeze(1)  # B*nch, 1, BW, T
            this_mask_imag_sum = this_mask_imag.sum(1).unsqueeze(1)  # B*nch, 1, BW, T
            this_mask_real = this_mask_real - (this_mask_real_sum - 1) / self.num_output
            this_mask_imag = this_mask_imag - this_mask_imag_sum / self.num_output
            est_spec_real = subband_spec[i].real.unsqueeze(1) * this_mask_real - subband_spec[i].imag.unsqueeze(1) * this_mask_imag  # B*nch, K, BW, T
            est_spec_imag = subband_spec[i].real.unsqueeze(1) * this_mask_imag + subband_spec[i].imag.unsqueeze(1) * this_mask_real  # B*nch, K, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        sep_subband_spec = torch.cat(sep_subband_spec, 2)
        
        output = torch.istft(sep_subband_spec.view(batch_size*nch*self.num_output, self.enc_dim, -1), 
                             n_fft=self.win, hop_length=self.stride,
                             window=torch.hann_window(self.win).to(input.device).type(input.type()), length=nsample)
        output = output.view(batch_size*nch, self.num_output, -1)
        # if was_one_d:
        #     return output.squeeze(0)
        return output
    
    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args