import scipy.io.wavfile as wav
import numpy as np
import tables as tb
from envelope import Envelope
from spectrogram import Spectrogram
from phonetic_data import PhoneData
from phonemes_and_features import PhoneInfo
from stimulus_matrix import StimMatrix
from response_data import Response
from scipy import signal
import matplotlib.pyplot as plt
from scipy import stats
# from cross_validation import CrossValidation

class Matrices:



    def construct_envelope_matrix(self, signal_data, time_array, current_freq, desired_freq, lag_min, lag_max):
        envelope = Envelope()
        envelope.calculate_envelope(signal_data)
        envelope.downsample_envelope(current_freq, desired_freq)
        normalised_envelope = envelope.normalise_envelope()
        normalised_envelope = np.asarray([normalised_envelope])
        # envelope.plot_envelope(signal_data, time_array)
        stimulus_matrix = StimMatrix()
        return stimulus_matrix.create_single_lag_matrix(normalised_envelope, desired_freq, lag_max)


    def construct_spectrogram_matrix(self, signal_data, time_array, current_freq, desired_freq, lag_min, lag_max):
        spectrogram = Spectrogram()
        spectrogram.calculate_frequency_bands()
        spectrogram.bandpass(signal_data, current_freq)
        # spectrogram.filtered_signals_plot(time_array, sampling_rate)
        spectrogram.hilbert_transform()

        spectrogram.downsample_spec(current_freq, desired_freq)

        spectrogram_envelopes = spectrogram.normalise_spec()
        stimulus_matrix = StimMatrix()
        return stimulus_matrix.create_single_lag_matrix(spectrogram_envelopes, desired_freq, lag_max)

        # spectrogram.spectrogram(time_array)

    def construct_phoneme_features_matrix(self,condition, trial, time_array, current_freq, desired_freq, lag_min, lag_max):
        phonetic_data = PhoneData()
        phonetic_data, features = phonetic_data.extract_json()

        phoneme_and_feature = PhoneInfo()
        phoneme_and_feature.phoneme_information(f'trials/{condition}_sentence_block_{trial}.json')
        phoneme_and_feature.calculate_var_array(time_array, phonetic_data, features)
        phoneme_and_feature.resample_phonemes_features(desired_freq, current_freq)


        stimulus_matrix = StimMatrix()
        return stimulus_matrix.create_single_lag_matrix(phoneme_and_feature.phoneme_array, desired_freq, lag_max)


    def construct_spectrogram_features_matrix(self, spectrogram, features):
        variable_array = np.array([[0] * features.shape[1]] * (features.shape[0] + 16))
        for index, envelope in enumerate(spectrogram):
            variable_array[index, :] = envelope

        variable_array[spectrogram.shape[0]:, :] = features
        return variable_array

    # def construct_all_matrices(self):

    def collect_variables(self, cnd, block, desired_freq, low_trim, high_trim):
        phonetic_data = PhoneData()
        phonetic_data, features = phonetic_data.extract_json()
        number_of_blocks = 30

        trial = f'trials/{cnd}_sentence_block_{block}'
        recording = f'{trial}.wav'
        sampling_rate, signal_data = wav.read(recording)
        signal_data = signal_data[int(low_trim * sampling_rate): int(high_trim * sampling_rate)]
        time_array = np.arange(0, len(signal_data)) / sampling_rate

        envelope = Envelope()
        envelope.calculate_envelope(signal_data)
        envelope.downsample_envelope(sampling_rate, desired_freq)
        normalised_envelope = envelope.normalise_envelope()
        normalised_envelope = np.asarray([normalised_envelope])
    # envelope.plot_envelope(signal_data, time_array)

        spectrogram = Spectrogram()
        spectrogram.calculate_frequency_bands()
        spectrogram.bandpass(signal_data, sampling_rate)
        # spectrogram.filtered_signals_plot(time_array, sampling_rate)
        spectrogram.hilbert_transform()

        spectrogram.downsample_spec(sampling_rate, desired_freq)

        spectrogram_envelopes = spectrogram.normalise_spec()
    # spectrogram.spectrogram(time_array)

        phoneme_and_feature = PhoneInfo()
        length = int(time_array.size * desired_freq / sampling_rate)
        resample = signal.resample(time_array, length)
        time_array = resample
        phoneme_and_feature.phoneme_information(f'{trial}.json')
        phoneme_and_feature.calculate_var_array(time_array, phonetic_data, features)
        # phoneme_and_feature.plot_multivar(time_array)

        return normalised_envelope, spectrogram_envelopes, phoneme_and_feature.phoneme_array, phoneme_and_feature.features_array

    def trim_normalise_eeg(self, data, max, trim, participants, blocks):
        single_ppt = data[0]
        single_ppt_trial = single_ppt[:,:,0]
        electrodes = single_ppt_trial.shape[0]
        length = single_ppt_trial[0, max: - trim].size

        normalised = np.zeros(shape=(participants,length, blocks, electrodes))

        for participant, participant_data in enumerate(data):
            for electrode in range(0, electrodes):
                for block in range(0, normalised.shape[2]):
                    recording = participant_data[electrode, max: - trim, block]
                    normalised[participant,:, block, electrode] = stats.zscore(recording)
        return normalised


    # def autocovariance(self, variable):
    #     auto = np.matmul(variable, variable.transpose())
    #         return auto
    #
    # def normalise_eeg(self, data, max, trim):
    #     length = data[0, max: - trim].size
    #     normalised = np.zeros(shape=(length, data.shape[0]))
    #     for electrode in range(0, data.shape[0]):
    #         minimum = np.amin(data[electrode, max: - trim])
    #         maximum = np.amax(data[electrode, max: - trim])
    #         stim = data[electrode, max: - trim]
    #         normalised[:, electrode] = stats.zscore(stim)
    #
    #     # for index, point in enumerate(stim):
    #     #     x = (point-minimum) / (maximum-minimum)
    #     #     normalised[index][electrode] = x
    #
    #         return normalised




if __name__ == '__main__':
    """note changed from load specific trial condition to just load specific condition as then not reading same data
    over and over just to extract one trial. was using  data = response.load_specific_condition_trial(index, block)
    
    still issues with this way... np.save sigkill because the array is much to large 
    trial_save = [stim_resp_mat(ppt_data, envelope_matrix, lag_max), autocovariance(envelope_matrix),
                                  stim_resp_mat(ppt_data, spectrogram_matrix, lag_max),
                                  autocovariance(spectrogram_matrix), stim_resp_mat(ppt_data, phoneme_matrix, lag_max),
                                  autocovariance(phoneme_matrix), stim_resp_mat(ppt_data, features_matrix, lag_max),
                                  autocovariance(features_matrix),
                                  stim_resp_mat(ppt_data, spectrogram_features_matrix, lag_max),
                                  autocovariance(spectrogram_features_matrix)]
                    print('2')
                    trial_save = np.asarray(trial_save)
                    print('3')
                    np.save(f'trial_matrix_data/{cnd}/trial_{block}_participant_{participant + 1}', trial_save)
                    print('4')
"""
    conditions = ('sense', 'nonsense', 'sense_control', 'nonsense_control')
    desired_freq = 1000
    # response = Response()
    # lag_min, lag_max = 0, 200
    # low_trim = 1.28
    # high_trim = 15.36
    # participants = 18
    # trials = 30
    # cv = CrossValidation(participants, trials)
    #
    # # for index, cnd in enumerate(conditions):
    # for index in range(0, 1):
    #     cnd = 'sense'
    #     data, skipped = response.load_specific_condition(index)
    #     data = trim_normalise_eeg(data, lag_max, int(low_trim * desired_freq), participants, trials)
    #
    #     for participant in range(0, participants):
    #         """had to trim and normalise all data first so could then use the responses in the prediction
    #         initially was only storing them in the loop below temporairly. Solution was to overwrite the
    #         original data"""
    #         if (participant + 1) not in skipped:
    #             ppt_data = data[participant, :, block, :]
    #
    #             for block in range(0, trials):
    #                 envelope, spectrogram, phonemes, features = collect_variables(cnd, block, desired_freq, low_trim, high_trim)
    #                 spectrogram_features = build_spectogram_features(spectrogram, features)
    #
    #                 stimulus_matrix = StimMatrix()
    #                 envelope_matrix = stimulus_matrix.create_single_lag_matrix(envelope, desired_freq, lag_max)
    #                 envelope_response = a = np.matmul(envelope_matrix, ppt_data)
    #                 envelope_auto = autocovariance(envelope_matrix)
    #                 cv.trial_parameter(envelope_auto, envelope_response, envelope_matrix, block, participant, ppt_data)
    #         # spectrogram_matrix = stimulus_matrix.create_single_lag_matrix(spectrogram, desired_freq, lag_max)
    #         # phoneme_matrix = stimulus_matrix.create_minmax_matrix(phonemes, desired_freq, lag_min, lag_max)
    #         # features_matrix = stimulus_matrix.create_minmax_matrix(features, desired_freq, lag_min, lag_max)
    #         # spectrogram_features_matrix = stimulus_matrix.create_minmax_matrix(spectrogram_features, desired_freq,
    #         #                                                                    lag_min, lag_max)
    #             cv.build_prediction(29, data, 0)





                    # spectrogram_response, normalised = covariance(ppt_data, spectrogram_matrix, lag_max, int(low_trim * desired_freq))
                    # spectrogram_auto = autocovariance(spectrogram_matrix)
                    # cv.trial_parameter(spectrogram_auto, spectrogram_response, spectrogram_matrix, block, participant, normalised)

                    #
                    #
                    # phoneme_response = stim_resp_mat(ppt_data, phoneme_matrix, lag_max)
                    # phoneme_auto = autocovariance(phoneme_matrix)
                    #
                    #
                    # features_response = stim_resp_mat(ppt_data, features_matrix, lag_max)
                    # features_auto = autocovariance(features_matrix)
                    #
                    #
                    # fs_response = stim_resp_mat(ppt_data, spectrogram_features_matrix, lag_max)
                    # fs_auto = autocovariance(spectrogram_features_matrix)

                    # trial_save = np.asarray(trial_save)
                    # np.save(f'trial_matrix_data/{cnd}/trial_{block}_participant_{participant + 1}', trial_save)


