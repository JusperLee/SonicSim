###
# Author: Kai Li
# Date: 2021-06-22 12:41:36
# LastEditors: Please set LastEditors
# LastEditTime: 2022-06-05 14:48:00
###
import csv
from sympy import im
import torch
import numpy as np
import logging
from pesq import pesq
from torchmetrics import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio
import fast_bss_eval
from scipy.linalg import toeplitz
# from .dnsmos import DNSMOS
from .sigmos import SigMOS
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

logger = logging.getLogger(__name__)

def is_silent(wav, threshold=1e-4):
    return torch.sum(wav ** 2) / wav.numel() < threshold

# ----------------------------- HELPERS ------------------------------------ #
def trim_mos(val):
    return min(max(val, 1), 5)

def lpcoeff(speech_frame, model_order):
    # (1) Compute Autocor lags
    winlength = speech_frame.shape[0]
    R = []
    for k in range(model_order + 1):
        first  = speech_frame[:(winlength - k)]
        second = speech_frame[k:winlength]
        R.append(np.sum(first * second))

    # (2) Lev-Durbin
    a = np.ones((model_order,))
    E = np.zeros((model_order + 1,))
    rcoeff = np.zeros((model_order,))
    E[0] = R[0]
    for i in range(model_order):
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
        rcoeff[i] = (R[i+1] - sum_term)/E[i]
        a[i] = rcoeff[i]
        if i > 0:
            a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
        E[i+1] = (1-rcoeff[i]*rcoeff[i])*E[i]
    acorr    = np.array(R, dtype=np.float32)
    refcoeff = np.array(rcoeff, dtype=np.float32)
    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    acorr    = np.array(acorr, dtype=np.float32)
    refcoeff = np.array(refcoeff, dtype=np.float32)
    lpparams = np.array(lpparams, dtype=np.float32)

    return acorr, refcoeff, lpparams
# -------------------------------------------------------------------------- #


def SSNR(ref_wav, deg_wav, srate=16000, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech     = ref_wav
    processed_speech = deg_wav
    clean_length     = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
    
    # scale both to have same dynamic range. Remove DC too.
    clean_speech     -= clean_speech.mean()
    processed_speech -= processed_speech.mean()
    processed_speech *= (np.max(np.abs(clean_speech)) / np.max(np.abs(processed_speech)))
   
    # Signal-to-Noise Ratio 
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) +
                                                        10e-20))
    # global variables
    winlength = int(np.round(30 * srate / 1000)) # 30 msecs
    skiprate  = winlength // 4
    MIN_SNR   = -10
    MAX_SNR   = 35

    # For each frame, calculate SSNR
    num_frames    = int(clean_length / skiprate - (winlength/skiprate))
    start         = 0
    time          = np.linspace(1, winlength, winlength) / (winlength + 1)
    window        = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame     = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame     = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy  = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps)+ eps))
        segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
        segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
        start += int(skiprate)
    return overall_snr, segmental_snr


def wss(ref_wav, deg_wav, srate):
    clean_speech     = ref_wav
    processed_speech = deg_wav
    clean_length     = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate  = np.floor(winlength / 4)
    max_freq  = srate / 2
    num_crit  = 25 # num of critical bands

    USE_FFT_SPECTRUM = 1
    n_fft    = int(2 ** np.ceil(np.log(2*winlength)/np.log(2)))
    n_fftby2 = int(n_fft / 2)
    Kmax     = 20
    Klocmax  = 1

    # Critical band filter definitions (Center frequency and BW in Hz)
    cent_freq = [50., 120, 190, 260, 330, 400, 470, 540, 617.372,
                 703.378, 798.717, 904.128, 1020.38, 1148.30, 
                 1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 
                 2211.08, 2446.71, 2701.97, 2978.04, 3276.17,
                 3597.63]
    bandwidth = [70., 70, 70, 70, 70, 70, 70, 77.3724, 86.0056,
                 95.3398, 105.411, 116.256, 127.914, 140.423, 
                 153.823, 168.154, 183.457, 199.776, 217.153, 
                 235.631, 255.255, 276.072, 298.126, 321.465,
                 346.136]

    bw_min = bandwidth[0] # min critical bandwidth

    # set up critical band filters. Note here that Gaussianly shaped filters
    # are used. Also, the sum of the filter weights are equivalent for each
    # critical band filter. Filter less than -30 dB and set to zero.
    min_factor = np.exp(-30. / (2 * 2.303)) # -30 dB point of filter

    crit_filter = np.zeros((num_crit, n_fftby2))
    all_f0 = []
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0.append(np.floor(f0))
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = list(range(n_fftby2))
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + \
                                   norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > \
                                                 min_factor)

    # For each frame of input speech, compute Weighted Spectral Slope Measure
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0 # starting sample
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compuet Power Spectrum of clean and processed
        clean_spec = (np.abs(np.fft.fft(clean_frame, n_fft)) ** 2)
        processed_spec = (np.abs(np.fft.fft(processed_frame, n_fft)) ** 2)
        clean_energy = [None] * num_crit
        processed_energy = [None] * num_crit

        # (3) Compute Filterbank output energies (in dB)
        for i in range(num_crit):
            clean_energy[i] = np.sum(clean_spec[:n_fftby2] * \
                                     crit_filter[i, :])
            processed_energy[i] = np.sum(processed_spec[:n_fftby2] * \
                                         crit_filter[i, :])
        clean_energy = np.array(clean_energy).reshape(-1, 1)
        eps = np.ones((clean_energy.shape[0], 1)) * 1e-10
        clean_energy = np.concatenate((clean_energy, eps), axis=1)
        clean_energy = 10 * np.log10(np.max(clean_energy, axis=1))
        processed_energy = np.array(processed_energy).reshape(-1, 1)
        processed_energy = np.concatenate((processed_energy, eps), axis=1)
        processed_energy = 10 * np.log10(np.max(processed_energy, axis=1))

        # (4) Compute Spectral Shape (dB[i+1] - dB[i])
        clean_slope = clean_energy[1:num_crit] - clean_energy[:num_crit-1]
        processed_slope = processed_energy[1:num_crit] - \
                processed_energy[:num_crit-1]

        # (5) Find the nearest peak locations in the spectra to each
        # critical band. If the slope is negative, we search
        # to the left. If positive, we search to the right.
        clean_loc_peak = []
        processed_loc_peak = []
        for i in range(num_crit - 1):
            if clean_slope[i] > 0:
                # search to the right
                n = i
                while n < num_crit - 1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak.append(clean_energy[n - 1])
            else:
                # search to the left
                n = i
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak.append(clean_energy[n + 1])
            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:
                n = i
                while n < num_crit - 1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak.append(processed_energy[n - 1])
            else:
                n = i
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak.append(processed_energy[n + 1])

        # (6) Compuet the WSS Measure for this frame. This includes
        # determination of the weighting functino
        dBMax_clean = max(clean_energy)
        dBMax_processed = max(processed_energy)

        # The weights are calculated by averaging individual
        # weighting factors from the clean and processed frame.
        # These weights W_clean and W_processed should range
        # from 0 to 1 and place more emphasis on spectral 
        # peaks and less emphasis on slope differences in spectral
        # valleys.  This procedure is described on page 1280 of
        # Klatt's 1982 ICASSP paper.
        clean_loc_peak = np.array(clean_loc_peak)
        processed_loc_peak = np.array(processed_loc_peak)
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit-1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - \
                                   clean_energy[:num_crit-1])
        W_clean = Wmax_clean * Wlocmax_clean
        Wmax_processed = Kmax / (Kmax + dBMax_processed - \
                                processed_energy[:num_crit-1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - \
                                      processed_energy[:num_crit-1])
        W_processed = Wmax_processed * Wlocmax_processed
        W = (W_clean + W_processed) / 2
        distortion.append(np.sum(W * (clean_slope[:num_crit - 1] - \
                                     processed_slope[:num_crit - 1]) ** 2))

        # this normalization is not part of Klatt's paper, but helps
        # to normalize the meaasure. Here we scale the measure by the sum of the
        # weights
        distortion[frame_count] = distortion[frame_count] / np.sum(W)
        start += int(skiprate)
    return distortion


def llr(ref_wav, deg_wav, srate):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    if srate < 10000:
        # LPC analysis order
        P = 10
    else:
        P = 16

    # For each frame of input speech, calculate the Log Likelihood Ratio
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Get the autocorrelation logs and LPC params used
        # to compute the LLR measure
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)
        A_clean = A_clean[None, :]
        A_processed = A_processed[None, :]

        # (3) Compute the LLR measure
        numerator = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
        denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)

        if (numerator/denominator) <= 0:
            print(f'Numerator: {numerator}')
            print(f'Denominator: {denominator}')

        log_ = np.log(numerator / denominator)
        distortion.append(np.squeeze(log_))
        start += int(skiprate)
    return np.nan_to_num(np.array(distortion))
# -------------------------------------------------------------------------- #

def PESQ_normalize(x):
    # Obtained from: https://github.com/nii-yamagishilab/NELE-GAN/blob/master/intel.py (def mapping_PESQ_harvard)
    a = -1.5
    b = 2.5
    y = 1/(1+np.exp(a *(x - b)))
    return y

# def PESQ_normalize(x):
#     y = (x + 0.5) / 5
#     return y

def CMOS_normalize(x):
    y = (x - 1.0) / 4
    return y

def compute_pesq(target_wav, pred_wav, fs, norm=False):
    # Compute the PESQ
    Pesq = pesq(fs, target_wav, pred_wav, 'wb')

    if norm:
        return PESQ_normalize(Pesq)
    else:
        return Pesq

def compute_csig(target_wav, pred_wav, fs, norm=False):
    alpha   = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(target_wav, pred_wav, 16000)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs     = LLR_dist
    LLR_len  = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, 'wb')

    # Csig
    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    Csig = float(trim_mos(Csig))
    
    if norm:
        return CMOS_normalize(Csig)
    else:
        return Csig

def compute_cbak(target_wav, pred_wav, fs, norm=False):
    alpha   = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute the SSNR
    snr_mean, segsnr_mean = SSNR(target_wav, pred_wav, 16000)
    segSNR = np.mean(segsnr_mean)

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, 'wb')

    # Cbak
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)

    if norm:
        return CMOS_normalize(Cbak)
    else:
        return Cbak

def compute_covl(target_wav, pred_wav, fs, norm=False):
    alpha   = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(target_wav, pred_wav, 16000)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs     = LLR_dist
    LLR_len  = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, 'wb')

    # Covl
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)

    if norm:
        return CMOS_normalize(Covl)
    else:
        return Covl

class MetricsTrackerNoASR:
    def __init__(self, save_file: str = ""):
        self.all_sdrs = []
        self.all_sdrs_i = []
        self.all_sisnrs = []
        self.all_sisnrs_i = []
        self.all_pesq_nb = []
        self.all_pesq_wb = []
        self.all_stoi = []
        self.all_mos_col = []
        self.all_mos_disc = []
        self.all_mos_loud = []
        self.all_mos_noise = []
        self.all_mos_reverb = []
        self.all_mos_sig = []
        self.all_mos_ovrl = []
        
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i", 'pesq_nb', 'pesq_wb', 'stoi', "MOS_COL", "MOS_DISC", "MOS_LOUD", "MOS_NOISE", "MOS_REVERB", "MOS_SIG", "MOS_OVRL"]
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=csv_columns)
        self.writer.writeheader()
        self.pesq_cal_nb = PerceptualEvaluationSpeechQuality(8000, "nb")
        self.pesq_cal_wb = PerceptualEvaluationSpeechQuality(16000, "wb")
        
        self.sigmos_cal = SigMOS("sigmos")
        
        
        self.sisdr_func = ScaleInvariantSignalDistortionRatio(zero_mean=True).cuda()
        self.sdr_func = SignalDistortionRatio(zero_mean=True).cuda()

    def __call__(self, mix, clean, estimate, key):
        if is_silent(clean):
            return
        # sisnr
        # import pdb; pdb.set_trace()
        if clean.shape[-1] < estimate.shape[-1]:
            estimate = estimate[:, :clean.shape[-1]]
        else:
            clean = clean[:, :estimate.shape[-1]]
            mix = mix[:, :estimate.shape[-1]]
            
        if clean.numel() == 0 or estimate.numel() == 0:
            import pdb; pdb.set_trace()
            
        sisnr = self.sisdr_func(estimate.unsqueeze(0), clean.unsqueeze(0))
        sisnr_baseline =self.sisdr_func(mix.unsqueeze(0), clean.unsqueeze(0))
        sisnr_i = sisnr - sisnr_baseline
        
        # sdr
        sdr = self.sdr_func(estimate.unsqueeze(0), clean.unsqueeze(0))
        sdr_baseline = self.sdr_func(mix.unsqueeze(0), clean.unsqueeze(0))
        sdr_i = sdr - sdr_baseline

        # PESQ
        pesq_nb = self.pesq_cal_nb(estimate.unsqueeze(0), clean.unsqueeze(0))
        pesq_wb = self.pesq_cal_wb(estimate.unsqueeze(0), clean.unsqueeze(0))
        
        # STOI
        stoi = short_time_objective_intelligibility(estimate.unsqueeze(0), clean.unsqueeze(0), 16000).float()
        # import pdb; pdb.set_trace()
        
        # Sigmos
        # import pdb; pdb.set_trace()
        sigmos_dicts = self.sigmos_cal.run(estimate.view(-1).cpu().numpy(), 16000)
        
        # import pdb; pdb.set_trace()
        
        row = {
            "snt_id": key + f"/s1.wav",
            "sdr": sdr.item(),
            "sdr_i": sdr_i.item(),
            "si-snr": sisnr.item(),
            "si-snr_i": sisnr_i.item(),
            'pesq_nb': pesq_nb.item(),
            'pesq_wb': pesq_wb.item(),
            'stoi': stoi.cpu().item(),
        }
        row.update(sigmos_dicts)
        
        self.writer.writerow(row)
        # Metric Accumulation
        self.all_sdrs.append(sdr.item())
        self.all_sdrs_i.append(sdr_i.item())
        self.all_sisnrs.append(sisnr.item())
        self.all_sisnrs_i.append(sisnr_i.item())
        self.all_pesq_nb.append(pesq_nb.item())
        self.all_pesq_wb.append(pesq_wb.item())
        self.all_stoi.append(stoi.cpu().item())
        self.all_mos_col.append(sigmos_dicts["MOS_COL"])
        self.all_mos_disc.append(sigmos_dicts["MOS_DISC"])
        self.all_mos_loud.append(sigmos_dicts["MOS_LOUD"])
        self.all_mos_noise.append(sigmos_dicts["MOS_NOISE"])
        self.all_mos_reverb.append(sigmos_dicts["MOS_REVERB"])
        self.all_mos_sig.append(sigmos_dicts["MOS_SIG"])
        self.all_mos_ovrl.append(sigmos_dicts["MOS_OVRL"])
    
    def update(self, ):
        return {"sdr": np.array(self.all_sdrs).mean(),
                "si-snr": np.array(self.all_sisnrs).mean(),
                "pesq_nb": np.array(self.all_pesq_nb).mean(),
                "pesq_wb": np.array(self.all_pesq_wb).mean(),
                "stoi": np.array(self.all_stoi).mean(),
                }

    def final(self,):
        row = {
            "snt_id": "avg",
            "sdr": np.array(self.all_sdrs).mean(),
            "sdr_i": np.array(self.all_sdrs_i).mean(),
            "si-snr": np.array(self.all_sisnrs).mean(),
            "si-snr_i": np.array(self.all_sisnrs_i).mean(),
            "pesq_nb": np.array(self.all_pesq_nb).mean(),
            "pesq_wb": np.array(self.all_pesq_wb).mean(),
            "stoi": np.array(self.all_stoi).mean(),
            "MOS_COL": np.array(self.all_mos_col).mean(),
            "MOS_DISC": np.array(self.all_mos_disc).mean(),
            "MOS_LOUD": np.array(self.all_mos_loud).mean(),
            "MOS_NOISE": np.array(self.all_mos_noise).mean(),
            "MOS_REVERB": np.array(self.all_mos_reverb).mean(),
            "MOS_SIG": np.array(self.all_mos_sig).mean(),
            "MOS_OVRL": np.array(self.all_mos_ovrl).mean(),
        }
        self.writer.writerow(row)
        row = {
            "snt_id": "std",
            "sdr": np.array(self.all_sdrs).std(),
            "sdr_i": np.array(self.all_sdrs_i).std(),
            "si-snr": np.array(self.all_sisnrs).std(),
            "si-snr_i": np.array(self.all_sisnrs_i).std(),
            "pesq_nb": np.array(self.all_pesq_nb).std(),
            "pesq_wb": np.array(self.all_pesq_wb).std(),
            "stoi": np.array(self.all_stoi).std(),
            "MOS_COL": np.array(self.all_mos_col).std(),
            "MOS_DISC": np.array(self.all_mos_disc).std(),
            "MOS_LOUD": np.array(self.all_mos_loud).std(),
            "MOS_NOISE": np.array(self.all_mos_noise).std(),
            "MOS_REVERB": np.array(self.all_mos_reverb).std(),
            "MOS_SIG": np.array(self.all_mos_sig).std(),
            "MOS_OVRL": np.array(self.all_mos_ovrl).std(),
        }
        self.writer.writerow(row)
        self.results_csv.close()
