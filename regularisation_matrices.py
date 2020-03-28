import scipy.io.wavfile as wav
import numpy as np
from envelope import Envelope
from spectrogram import Spectrogram
from phonetic_data import PhoneData
from phonemes_and_features import PhoneInfo
from stimulus_matrix import StimMatrix
from response_data import Response

conditions = ('sense', 'nonsense', 'sense_control', 'nonsense_control')
envelope_mat = np.ndarray(shape=(30, len(conditions)), dtype=np.ndarray)
spectrogram_mat = np.ndarray(shape=(30, len(conditions)), dtype=np.ndarray)
phoneme_mat = np.ndarray(shape=(30, len(conditions)), dtype=np.ndarray)
feature_mat = np.ndarray(shape=(30, len(conditions)), dtype=np.ndarray)
env_mat_ = np.ndarray(shape=(30, len(conditions)), dtype=np.ndarray)

phonetic_data = PhoneData()
phonetic_data, features = phonetic_data.extract_json()
number_of_blocks = 29
first_sample_time = 1.28

for cnd in conditions:
    for block in range(1, number_of_blocks + 1):
        trial = f'trials/{cnd}_sentence_block_{block}'
        recording = f'{trial}.wav'
        sampling_rate, signal_data = wav.read(recording)
        time_array = np.arange(0, len(signal_data)) / sampling_rate

        envelope = Envelope()
        envelope.calculate_envelope(signal_data)
        # envelope.plot_envelope(signal_data, time_array)

        spectrogram = Spectrogram()
        spectrogram.calculate_frequency_bands()
        spectrogram.bandpass(signal_data, sampling_rate)
        # spectrogram.filtered_signals_plot(time_array, sampling_rate)
        spectrogram.hilbert_transform()
        # spectrogram.spectrogram(time_array)

        phoneme_and_feature = PhoneInfo()
        print(trial)
        phoneme_and_feature.phoneme_information(f'{trial}.json')
        phoneme_and_feature.calculate_var_array(time_array, phonetic_data, features)
        # phoneme_and_feature.plot_multivar(time_array)

        envelope.downsample_envelope(sampling_rate)
        normalised_envelope = envelope.normalise_envelope()
        normalised_envelope = np.asarray([normalised_envelope])
        stimulus_matrix = StimMatrix()
        stim_mat = stimulus_matrix.create_minmax_matrix(normalised_envelope, 1000, first_sample_time)
        autocovariance = np.matmul(stim_mat.transpose(), stim_mat)

        for ppt in range(1, 19):
            response = Response()
            response_data = response.read_data(cnd, ppt)
            resp = response_data[0][:][block]
            s_t_r = stim_mat


















