import numpy as np
import torch
import torchaudio
from matplotlib import pyplot as plt


def run_model(model, inputs, file_duration, detection_thresh, spec_width, scale_factor, nms_win_size, local_batch_size):
    iters = int(np.ceil(len(inputs) / local_batch_size))
    y_prediction = []
    for i in range(iters):
        y = model(inputs[local_batch_size * i:local_batch_size * (i + 1)])
        y = y.softmax(axis=-1)
        y = y.to('cpu').detach().numpy().astype(np.float64)
        y_prediction.append(y)

    y_prediction = np.concatenate(y_prediction, axis=0)
    y_prediction = y_prediction[:, 1]
    pos, prob = nms_1d(y_prediction.astype('float'), nms_win_size / scale_factor, file_duration, spec_width,
                       scale_factor)
    return pos[prob[:, 0] > detection_thresh], prob[prob > detection_thresh], y_prediction


def nms_1d(src, win_size, file_duration, spec_width, scale_factor):
    pos = []
    src_cnt = 0
    max_ind = 0
    ii = 0
    ee = 0
    width = src.shape[0] - 1
    while ii <= width:

        if max_ind < (ii - win_size):
            max_ind = ii - win_size

        ee = np.minimum(ii + win_size, width)

        while max_ind <= ee:
            src_cnt += 1
            if src[int(max_ind)] > src[int(ii)]:
                break
            max_ind += 1

        if max_ind > ee:
            pos.append(ii)
            max_ind = ii + 1
            ii += win_size

        ii += 1

    pos = np.asarray(pos).astype('int')
    val = src[pos]

    inds = (pos + win_size) < src.shape[0]
    pos = pos[inds]
    val = val[inds]

    pos = (pos * scale_factor * 1.) / spec_width
    pos = pos * file_duration

    return pos, val[..., np.newaxis]


def visualize_calls(filepath, det_dict, audio_conf, plot=True):
    specs = []
    waveform, sr = torchaudio.load(filepath)
    for i, t in enumerate(det_dict['det_time']):
        s = max(0, t - 0.03)
        call = waveform[:, int(s * sr): int((t + 0.3) * sr)]
        spec = torchaudio.compliance.kaldi.fbank(call,
                                                 htk_compat=True,
                                                 sample_frequency=sr,
                                                 use_energy=False,
                                                 window_type='hanning',
                                                 num_mel_bins=128,
                                                 dither=0.0,
                                                 frame_shift=(audio_conf['fft_win_length'] * 1000) - (
                                                         audio_conf['fft_win_length'] * 1000) * audio_conf[
                                                                 'fft_overlap'],
                                                 frame_length=audio_conf['fft_win_length'] * 1000,
                                                 high_freq=12000,
                                                 low_freq=500)
        spec = torch.flipud(spec.T).numpy()

        if plot:
            plt.imshow(spec, cmap='gray')
            plt.grid(False)
            plt.title("prob={:.3f}  time={:.2f}".format(det_dict['det_prob'][i], t.numpy()))
            plt.show()
        specs.append(spec)
    return specs
