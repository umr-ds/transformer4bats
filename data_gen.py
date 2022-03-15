import torchaudio
import torch
import torch.nn.functional
from torch.utils.data import Dataset
from skimage.util.shape import view_as_windows
import os


class DataGen(Dataset):
    def __init__(self, files, raw_audio_dir, audio_conf):

        self.files = files
        self.raw_audio_dir = raw_audio_dir
        self.audio_conf = audio_conf
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.denoise = self.audio_conf.get('denoise')
        self.time_expand = self.audio_conf.get('time_expand')
        self.target_sample_rate = self.audio_conf.get('target_sample_rate')
        self.label_num = 2
        self.frame_length = self.audio_conf.get('frame_length') if self.audio_conf.get('frame_length') else 25
        self.frame_shift = self.audio_conf.get('frame_shift') if self.audio_conf.get('frame_shift') else 5
        self.fft_overlap = self.audio_conf.get('fft_overlap')  # 0.843 #%
        self.fft_win_length = self.audio_conf.get('fft_win_length')  # 0.023 #s
        self.window_width = self.audio_conf.get('window_width')
        self.slide_window_stride = self.audio_conf.get('slide_window_stride')

        self.x = files

    def wav2melspec(self, filename):
        sig, sr = torchaudio.load(filename)
        if self.time_expand:
            sr = sr / 10
        if self.target_sample_rate is not None and self.target_sample_rate != sr:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            sig = resample(sig)
            sr = self.target_sample_rate
        sig = sig - sig.mean()

        spec = torchaudio.compliance.kaldi.fbank(sig,
                                                 htk_compat=True,
                                                 sample_frequency=sr,
                                                 use_energy=False,
                                                 window_type='hanning',
                                                 num_mel_bins=self.melbins,
                                                 dither=0.0,
                                                 frame_shift=(self.fft_win_length * 1000) - (
                                                             self.fft_win_length * 1000) * self.fft_overlap,
                                                 frame_length=self.fft_win_length * 1000,
                                                 high_freq=12000,
                                                 low_freq=500)

        spec = torch.transpose(spec, 0, 1).numpy()
        local_feats_wins = view_as_windows(spec, (spec.shape[0], self.window_width), self.slide_window_stride)[0]
        spec_width = spec.shape[1]

        return local_feats_wins, len(sig[0]) / sr, spec_width

    def __getitem__(self, index):
        spec, duration, spec_width = self.wav2melspec(os.path.join(self.raw_audio_dir, self.x[index]))
        spec = torch.from_numpy(spec)
        spec = torch.transpose(spec, 1, 2)
        spec = (spec - self.norm_mean) / (self.norm_std * 2)
        return spec, duration, spec_width, self.x[index]

    def __len__(self):
        return len(self.x)
