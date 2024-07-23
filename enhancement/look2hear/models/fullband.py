import torch
from torch.nn import functional
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from .base_model import BaseModel

EPSILON = np.finfo(np.float32).eps

def stft(y, n_fft, hop_length, win_length):
    """Wrapper of the official torch.stft for single-channel and multi-channel.

    Args:
        y: single- or multi-channel speech with shape of [B, C, T] or [B, T]
        n_fft: number of FFT
        hop_length: hop length
        win_length: hanning window size

    Shapes:
        mag: [B, F, T] if dims of input is [B, T], whereas [B, C, F, T] if dims of input is [B, C, T]

    Returns:
        mag, phase, real and imag with the same shape of [B, F, T] (**complex-valued** STFT coefficients)
    """
    num_dims = y.dim()
    assert num_dims == 2 or num_dims == 3, "Only support 2D or 3D Input"

    batch_size = y.shape[0]
    num_samples = y.shape[-1]

    if num_dims == 3:
        y = y.reshape(-1, num_samples)  # [B * C ,T]

    complex_stft = torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=y.device),
        return_complex=True,
    )
    _, num_freqs, num_frames = complex_stft.shape

    if num_dims == 3:
        complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, num_frames)

    mag = torch.abs(complex_stft)
    phase = torch.angle(complex_stft)
    real = complex_stft.real
    imag = complex_stft.imag
    return mag, phase, real, imag

class SequenceModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        bidirectional,
        sequence_model="GRU",
        output_activate_function="Tanh",
    ):
        """
        Wrapper of conventional sequence models (LSTM or GRU)

        Args:
            input_size: input size.
            output_size: when projection_size> 0, the linear layer is used for projection. Otherwise, no linear layer.
            hidden_size: hidden size.
            num_layers:  number of layers.
            bidirectional: whether to use bidirectional RNN.
            sequence_model: LSTM | GRU.
            output_activate_function: Tanh | ReLU | ReLU6 | LeakyReLU | PReLU | None.
        """
        super().__init__()
        # Sequence layer
        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "SRU":
            pass
            # self.sequence_model = CustomSRU(
            #     input_size=input_size,
            #     hidden_size=hidden_size,
            #     num_layers=num_layers,
            #     bidirectional=bidirectional,
            #     highway_bias=-2
            # )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if int(output_size):
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {self.activate_function}"
                )

        self.output_activate_function = output_activate_function
        self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3, f"The shape of input is {x.shape}."
        self.sequence_model.flatten_parameters()

        x = x.permute(0, 2, 1)  # [B, F, T] => [B, T, F]
        o, _ = self.sequence_model(x)

        if self.output_size:
            o = self.fc_output_layer(o)

        if self.output_activate_function:
            o = self.activate_function(o)
        o = o.permute(0, 2, 1)  # [B, T, F] => [B, F, T]
        return o



class Fullband(BaseModel):
    def __init__(
        self,
        num_freqs,
        hidden_size,
        sequence_model,
        output_activate_function,
        look_ahead,
        n_fft,
        hop_length,
        win_length,
        norm_type="offline_laplace_norm",
        weight_init=True,
        sample_rate=16000
    ):
        """Fullband Model (cIRM mask)

        Args:
            num_freqs:
            hidden_size:
            sequence_model:
            output_activate_function:
            look_ahead:
        """
        super().__init__(sample_rate=sample_rate)
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        self.fullband_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs * 2,
            hidden_size=hidden_size,
            num_layers=3,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=output_activate_function,
        )

        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        if weight_init:
            print("Initializing model...")
            self.apply(self.weight_init)
        
    @staticmethod
    def freq_unfold(input, num_neighbors):
        """Split the overlapped subband units along the frequency axis.

        Args:
            input: four-dimension input with the shape [B, C, F, T]
            num_neighbors: number of neighbors in each side for each subband unit.

        Returns:
            Overlapped sub-band units specified as [B, N, C, F_s, T], where `F_s` represents the frequency axis of
            each sub-band unit and `N` is the number of sub-band unit, e.g., [2, 161, 1, 19, 100].
        """
        assert input.dim() == 4, f"The dim of the input is {input.dim()}. It should be four dim."
        batch_size, num_channels, num_freqs, num_frames = input.size()

        if num_neighbors <= 0:  # No change to the input
            return input.permute(0, 2, 1, 3).reshape(batch_size, num_freqs, num_channels, 1, num_frames)

        output = input.reshape(batch_size * num_channels, 1, num_freqs, num_frames)  # [B * C, 1, F, T]
        sub_band_unit_size = num_neighbors * 2 + 1

        # Pad the top and bottom of the original spectrogram
        output = functional.pad(output, [0, 0, num_neighbors, num_neighbors], mode="reflect")  # [B * C, 1, F, T]

        # Unfold the spectrogram into sub-band units
        # [B * C, 1, F, T] => [B * C, sub_band_unit_size, num_frames, N], N is equal to the number of frequencies.
        output = functional.unfold(output, kernel_size=(sub_band_unit_size, num_frames))  # move on the F and T axes
        assert output.shape[-1] == num_freqs, f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"

        # Split the dimension of the unfolded feature
        output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()  # [B, N, C, F_s, T]

        return output

    @staticmethod
    def _reduce_complexity_separately(sub_band_input, full_band_output, device):
        """
        Group Dropout for FullSubNet

        Args:
            sub_band_input: [60, 257, 1, 33, 200]
            full_band_output: [60, 257, 1, 3, 200]
            device:

        Notes:
            1. 255 and 256 freq not able to be trained
            2. batch size should can be divided by 3, or the last batch will be dropped

        Returns:
            [60, 85, 1, 36, 200]
        """
        batch_size = full_band_output.shape[0]
        n_freqs = full_band_output.shape[1]
        sub_batch_size = batch_size // 3
        final_selected = []

        for idx in range(3):
            # [0, 60) => [0, 20)
            sub_batch_indices = torch.arange(
                idx * sub_batch_size, (idx + 1) * sub_batch_size, device=device
            )
            full_band_output_sub_batch = torch.index_select(
                full_band_output, dim=0, index=sub_batch_indices
            )
            sub_band_output_sub_batch = torch.index_select(
                sub_band_input, dim=0, index=sub_batch_indices
            )

            # Avoid to use padded value (first freq and last freq)
            # i = 0, (1, 256, 3) = [1, 4, ..., 253]
            # i = 1, (2, 256, 3) = [2, 5, ..., 254]
            # i = 2, (3, 256, 3) = [3, 6, ..., 255]
            freq_indices = torch.arange(idx + 1, n_freqs - 1, step=3, device=device)
            full_band_output_sub_batch = torch.index_select(
                full_band_output_sub_batch, dim=1, index=freq_indices
            )
            sub_band_output_sub_batch = torch.index_select(
                sub_band_output_sub_batch, dim=1, index=freq_indices
            )

            # ([30, 85, 1, 33 200], [30, 85, 1, 3, 200]) => [30, 85, 1, 36, 200]

            final_selected.append(
                torch.cat([sub_band_output_sub_batch, full_band_output_sub_batch], dim=-2)
            )

        return torch.cat(final_selected, dim=0)

    @staticmethod
    def forgetting_norm(input, sample_length=192):
        """
        Using the mean value of the near frames to normalization

        Args:
            input: feature
            sample_length: length of the training sample, used for calculating smooth factor

        Returns:
            normed feature

        Shapes:
            input: [B, C, F, T]
            sample_length_in_training: 192
        """
        assert input.ndim == 4
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size, num_channels * num_freqs, num_frames)

        eps = 1e-10
        mu = 0
        alpha = (sample_length - 1) / (sample_length + 1)

        mu_list = []
        for frame_idx in range(num_frames):
            if frame_idx < sample_length:
                alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(
                    input[:, :, frame_idx], dim=1
                ).reshape(
                    batch_size, 1
                )  # [B, 1]
            else:
                current_frame_mu = torch.mean(input[:, :, frame_idx], dim=1).reshape(
                    batch_size, 1
                )  # [B, 1]
                mu = alpha * mu + (1 - alpha) * current_frame_mu

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        output = input / (mu + eps)

        output = output.reshape(batch_size, num_channels, num_freqs, num_frames)
        return output

    @staticmethod
    def hybrid_norm(input, sample_length_in_training=192):
        """
        Args:
            input: [B, F, T]
            sample_length_in_training:

        Returns:
            [B, F, T]
        """
        assert input.ndim == 3
        device = input.device
        data_type = input.dtype
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10

        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)
        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(
                    batch_size, 1
                )  # [B, 1]
                mu_list.append(mu)
            else:
                break
        initial_mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]

        step_sum = torch.sum(input, dim=1)  # [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device
        )
        entry_count = entry_count.reshape(1, n_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cum_mean = cumulative_sum / entry_count  # B, T

        cum_mean = cum_mean.reshape(batch_size, 1, n_frames)  # [B, 1, T]

        # print(initial_mu[0, 0, :50])
        # print("-"*60)
        # print(cum_mean[0, 0, :50])
        cum_mean[:, :, :sample_length_in_training] = initial_mu

        return input / (cum_mean + eps)

    @staticmethod
    def offline_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        # utterance-level mu
        mu = torch.mean(input, dim=list(range(1, input.dim())), keepdim=True)

        normed = input / (mu + 1e-5)

        return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:

        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device,
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # B, T
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)

        normed = input / (cumulative_mean + EPSILON)

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def drop_band(input, num_groups=2):
        """Reduce computational complexity of the sub-band part in the FullSubNet model.

        Shapes:
            input: [B, C, F, T]
            return: [B, C, F // num_groups, T]
        """
        batch_size, _, num_freqs, _ = input.shape
        assert (
            batch_size > num_groups
        ), f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

        if num_groups <= 1:
            # No demand for grouping
            return input

        # Each sample must has the same number of the frequencies for parallel training.
        # Therefore, we need to drop those remaining frequencies in the high frequency part.
        if num_freqs % num_groups != 0:
            input = input[..., : (num_freqs - (num_freqs % num_groups)), :]
            num_freqs = input.shape[2]

        output = []
        for group_idx in range(num_groups):
            samples_indices = torch.arange(
                group_idx, batch_size, num_groups, device=input.device
            )
            freqs_indices = torch.arange(
                group_idx, num_freqs, num_groups, device=input.device
            )

            selected_samples = torch.index_select(input, dim=0, index=samples_indices)
            selected = torch.index_select(
                selected_samples, dim=2, index=freqs_indices
            )  # [B, C, F // num_groups, T]

            output.append(selected)

        return torch.cat(output, dim=0)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.std(input, dim=(1, 2, 3), keepdim=True)

        normed = (input - mu) / (std + 1e-5)

        return normed

    @staticmethod
    def cumulative_layer_norm(input):
        """
        Online zero-norm

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        step_pow_sum = torch.sum(torch.square(input), dim=1)

        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
        cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device,
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # [B, T]
        cumulative_var = (
            cumulative_pow_sum - 2 * cumulative_mean * cumulative_sum
        ) / entry_count + cumulative_mean.pow(
            2
        )  # [B, T]
        cumulative_std = torch.sqrt(cumulative_var + EPSILON)  # [B, T]

        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)
        cumulative_std = cumulative_std.reshape(batch_size * num_channels, 1, num_frames)

        normed = (input - cumulative_mean) / cumulative_std

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            norm = self.offline_laplace_norm
        elif norm_type == "cumulative_laplace_norm":
            norm = self.cumulative_laplace_norm
        elif norm_type == "offline_gaussian_norm":
            norm = self.offline_gaussian_norm
        elif norm_type == "cumulative_layer_norm":
            norm = self.cumulative_layer_norm
        elif norm_type == "forgetting_norm":
            norm = self.forgetting_norm
        else:
            raise NotImplementedError(
                "You must set up a type of Norm. "
                "e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc."
            )
        return norm

    def weight_init(self, m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

    def forward(self, noisy):
        """
        Args:
            noisy_mag: [B, 1, F, T], noisy magnitude spectrogram

        Returns:
            [B, 2, F, T], the real part and imag part of the enhanced spectrogram
        """
        noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        noisy_mag = noisy_mag.unsqueeze(1)    
        
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert (
            num_channels == 1
        ), f"{self.__class__.__name__} takes the mag feature as inputs."

        noisy_mag = self.norm(noisy_mag).reshape(
            batch_size, num_channels * num_freqs, num_frames
        )
        output = self.fullband_model(noisy_mag).reshape(
            batch_size, 2, num_freqs, num_frames
        )

        return output[:, :, :, self.look_ahead :], noisy_real, noisy_imag

    def get_model_args(self):
        model_args = {"n_src": 1}
        return model_args