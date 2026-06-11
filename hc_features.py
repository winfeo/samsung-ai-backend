import numpy as np
import torch
import torchaudio.transforms as T

SAMPLE_RATE = 48000

def extract_hc_gpu_enhanced(waveform_48khz):
    if waveform_48khz.dim() == 2:
        waveform_48khz = waveform_48khz.squeeze(0)
    y = waveform_48khz
    dev = y.device

    rms = torch.sqrt(torch.mean(y ** 2))
    rms_std = torch.sqrt(torch.mean((y - torch.mean(y)) ** 2))
    
    segment_size = len(y) // 10
    rms_segments = torch.tensor([torch.sqrt(torch.mean(y[i*segment_size:(i+1)*segment_size] ** 2)) for i in range(10)], device=dev)
    rms_variation = torch.std(rms_segments) / (torch.mean(rms_segments) + 1e-7)

    zcr = torch.mean((torch.diff(torch.sign(y)) != 0).float())
    zcr_segments = torch.tensor([torch.mean((torch.diff(torch.sign(y[i*segment_size:(i+1)*segment_size])) != 0).float()) for i in range(10)], device=dev)
    zcr_variation = torch.std(zcr_segments) / (torch.mean(zcr_segments) + 1e-7)

    n_fft, hop_length = 2048, int(SAMPLE_RATE * 0.01)
    window = torch.hann_window(n_fft, device=dev)
    spec = torch.stft(y, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    mag = torch.abs(spec)

    freqs = torch.arange(mag.shape[0], device=dev, dtype=torch.float32) * SAMPLE_RATE / n_fft
    centroid_per_frame = (freqs.unsqueeze(1) * mag).sum(dim=0) / (mag.sum(dim=0) + 1e-7)
    centroid = centroid_per_frame.mean()
    centroid_std = centroid_per_frame.std()

    centroid_expanded = centroid.unsqueeze(0).unsqueeze(1)
    freqs_expanded = freqs.unsqueeze(1)
    bandwidth = torch.sqrt(torch.sum(((freqs_expanded - centroid_expanded) ** 2) * mag) / (torch.sum(mag) + 1e-7))

    cumsum = torch.cumsum(torch.sum(mag, dim=1), dim=0)
    total_energy = cumsum[-1]
    rolloff = freqs[min(torch.searchsorted(cumsum, 0.85 * total_energy), len(freqs)-1)]
    rolloff95 = freqs[min(torch.searchsorted(cumsum, 0.95 * total_energy), len(freqs)-1)]

    mfcc_transform = T.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=5, melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': 40}).to(dev)
    mfcc = mfcc_transform(y.unsqueeze(0))
    mfcc_means = mfcc.mean(dim=2).squeeze(0)
    mfcc_stds = mfcc.std(dim=2).squeeze(0)

    onset_env = torch.diff(mag.sum(dim=0)).clamp(min=0).cpu().numpy()
    ac = np.correlate(onset_env, onset_env, mode='full'); ac = ac[len(ac)//2:]
    min_period, max_period = int(SAMPLE_RATE / (200/60) / hop_length), int(SAMPLE_RATE / (50/60) / hop_length)
    if max_period > len(ac): max_period = len(ac) - 1
    if min_period < max_period:
        peak_idx = np.argmax(ac[min_period:max_period]) + min_period
        tempo = 60 * SAMPLE_RATE / (hop_length * peak_idx)
        rhythm_strength = ac[peak_idx] / (ac.mean() + 1e-7)
    else:
        tempo, rhythm_strength = 120.0, 1.0

    chroma = mag[:12, :].mean()
    chroma_std = mag[:12, :].mean(dim=0).std()

    spec_flat = mag.mean(dim=1)
    spectral_contrast = (spec_flat.max() - spec_flat.min()) / (spec_flat.mean() + 1e-7)
    harmonic_ratio = (spec_flat.max() / (spec_flat.mean() + 1e-7)).clamp(0, 10) / 10

    bass_energy = mag[:20, :].mean() / (mag.mean() + 1e-7)
    mid_energy = mag[20:60, :].mean() / (mag.mean() + 1e-7)
    treble_energy = mag[60:, :].mean() / (mag.mean() + 1e-7)

    seg_len = len(y) // 5
    tempos_seg, rhythm_peaks = [], []
    for i in range(5):
        seg = y[i*seg_len:(i+1)*seg_len]
        onset_env_seg = torch.diff(torch.abs(torch.stft(seg, n_fft=2048, hop_length=512, window=torch.hann_window(2048, device=dev), return_complex=True)).sum(dim=0)).clamp(min=0).cpu().numpy()
        ac_seg = np.correlate(onset_env_seg, onset_env_seg, mode='full'); ac_seg = ac_seg[len(ac_seg)//2:]
        if max_period > len(ac_seg): max_period = len(ac_seg) - 1
        if min_period < max_period:
            peak_idx_seg = np.argmax(ac_seg[min_period:max_period]) + min_period
            tempos_seg.append(60 * SAMPLE_RATE / (512 * peak_idx_seg))
            rhythm_peaks.append(ac_seg[peak_idx_seg] / (ac_seg.mean() + 1e-7))
        else:
            tempos_seg.append(120.0); rhythm_peaks.append(1.0)
    tempos_seg = np.array(tempos_seg)
    rhythm_peaks = np.array(rhythm_peaks)

    return np.array([
        float(tempo), float(rms.cpu()), float(rms_std.cpu()),
        float(centroid.cpu()), float(bandwidth.cpu()), float(rolloff.cpu()),
        float(rms_variation.cpu()), float(zcr_variation.cpu()),
        float(centroid_std.cpu()), float(chroma_std.cpu()), float(rhythm_strength),
        float(rolloff95.cpu()), float(spectral_contrast.cpu()),
        float(harmonic_ratio.cpu()), float(zcr.cpu()),
        float(mfcc_means[0].cpu()), float(mfcc_means[1].cpu()),
        float(mfcc_means[2].cpu()), float(mfcc_means[3].cpu()), float(mfcc_means[4].cpu()),
        float(mfcc_stds[0].cpu()), float(mfcc_stds[1].cpu()), float(mfcc_stds[2].cpu()),
        float(mfcc_stds[3].cpu()), float(mfcc_stds[4].cpu()),
        float(bass_energy.cpu()), float(mid_energy.cpu()), float(treble_energy.cpu()),
        float(chroma.cpu()),
        float(np.std(tempos_seg) / (np.mean(tempos_seg) + 1e-7)),
        float((np.max(tempos_seg) - np.min(tempos_seg)) / (np.mean(tempos_seg) + 1e-7)),
        float(np.mean(rhythm_peaks)),
        float(np.std(rhythm_peaks) / (np.mean(rhythm_peaks) + 1e-7)),
    ], dtype=np.float32)
