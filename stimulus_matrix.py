import numpy as np
import scipy.io.wavfile as wav
from envelope import Envelope
from spectrogram import Spectrogram
from phonetic_data import PhoneData
from phonemes_and_features import PhoneInfo
from numpy.linalg import inv


class StimMatrix:

    def create_minmax_matrix(self, stimulus, sample_rate, minimum, maximum):
        """creates time lagged matrix, with each row associated with the time window influencing response at time t.
        for multi-variable matrices multiple rows will contain information corresponding to the same response time.
        the window is derived from two time points which are earlier than the current time point.

        arguments and variables:
        stimulus: np.array containing either envelope, spectrogram, phoneme or features information
        sample_rate: the sampling rate being used - this will be the downsampled value 1000 Hz
        first_sample: initial recording point in the time series
        variables: establishes whether the matrix is a product of one variable or multiple.
        minimum: stimulus index corresponding to the minimum lag point
        maximum: stimulus index corresponding to the maximum lag point
        lag: size of the time lag window
        time: as the first response is not a t = 0, time reflects the length of the time series.
        matrix: contains the matrix of all time lagged windows [lag][time]. initialised as an 2D np.array of zeroes.
        start: start of time array

        returns:
        the created time lagged stimulus matrix
        """

        variables = stimulus.shape[0]

        minimum = int((minimum/1000) * sample_rate)
        maximum = int((maximum/1000) * sample_rate)
        start = maximum

        lag = (start - minimum) - (start - maximum) + 1

        time = stimulus[0][start - minimum: - minimum].size

        matrix = np.zeros(shape=(variables * lag, time))
        count = 0

        while count < matrix.shape[0]:
            new_count = count + variables
            rows = stimulus[:, start - minimum: - minimum]
            matrix[count: new_count] = rows
            minimum += 1
            count = new_count

        return matrix


    def create_single_lag_matrix(self, stimulus, sample_rate, lag_point):
        """creates time lagged matrix, with each row associated with the time window influencing response at time t.
        for multi-variable matrices multiple rows will contain information corresponding to the same response time.
        the window derived from all times before the current time point to the specified distance back

        arguments and variables:
        stimulus: np.array containing either envelope, spectrogram, phoneme or features information
        sample_rate: the sampling rate being used - this will be the downsampled value 1000 Hz
        first_sample: initial recording point in the time series
        variables: establishes whether the matrix is a product of one variable or multiple.
        lag_point: stimulus index corresponding time furthest back being assessed
        lag: size of the time lag window
        time: as the first response is not a t = 0, time reflects the length of the time series.
        matrix: contains the matrix of all time lagged windows [lag][time]. initialised as an 2D np.array of zeroes.
        start: start of time array

        returns:
        the time lagged stimulus matrix
        """
        variables = stimulus.shape[0]
        lag_point = int((lag_point / 1000) * sample_rate)
        lag = lag_point + 1
        start = lag_point

        time = stimulus[0][start:].size


        matrix = np.zeros(shape=(time, lag * variables))

        lag_step = 0

        count = 0
        while count < time:
            new_count = count + variables
            for variable in range(0, variables):
                matrix[count, variable: variable + lag] = stimulus[variable, start - 200: start + 1]
            count += 1
            start += 1

        return matrix


def get_stimulus(self, stimulus, requested_sample_rate, first_sample):
        """ builds and returns the stimulus matrix and the modified identity matrix
        """

        stim_matrix = self.create_minmax_matrix(stimulus, requested_sample_rate, first_sample)
        autocovariance = np.matmul(stim_matrix.transpose(), stim_matrix)
        inverse = inv(autocovariance)
        smoothing = self.smoothing_matrix(inverse.shape[0])

        return stim_matrix, inverse, smoothing

if __name__ == '__main__':

    low_trim = 1.28
    high_trim = 50.72

    trial = f'trials/sense_sentence_block_0'
    recording = f'{trial}.wav'
    sampling_rate, signal_data = wav.read(recording)
    signal_data = signal_data[int(low_trim * sampling_rate): int(high_trim * sampling_rate)]
    time_array = np.arange(0, len(signal_data)) / sampling_rate
    desired_freq = 1000

    envelope = Envelope()
    envelope.calculate_envelope(signal_data)
    print(envelope.envelope.shape)
    envelope.downsample_envelope(sampling_rate, desired_freq)
    normalised_envelope = envelope.normalise_envelope()
    normalised_envelope = np.asarray([normalised_envelope])
    print(normalised_envelope.shape)
    sm = StimMatrix()
    # matrix = sm.create_minmax_matrix(normalised_envelope, 1000)
    # print(matrix.shape)

    spectrogram = Spectrogram()
    spectrogram.calculate_frequency_bands()
    spectrogram.bandpass(signal_data, sampling_rate)
    spectrogram.hilbert_transform()
    spectrogram.downsample_spec(sampling_rate, desired_freq)
    envelopes = spectrogram.normalise_spec()
    # matrix = sm.create_minmax_matrix(envelopes, 1000)
    # print(matrix.shape)


    phonetic_data = PhoneData()
    phonetic_data, features = phonetic_data.extract_json()

    phoneme_and_feature = PhoneInfo()

    time_array = np.arange(0, int(envelope.envelope.size)) / 1000
    phoneme_and_feature.phoneme_information(f'{trial}.json')
    phoneme_and_feature.calculate_var_array(time_array, phonetic_data, features)

    # sm.create_minmax_matrix(phoneme_and_feature.phoneme_array, 1000)
    # sm.create_minmax_matrix(phoneme_and_feature.features_array, 1000)
    sm.create_single_lag_matrix(phoneme_and_feature.features_array, 1000, 200)



