import numpy as np
import scipy.io.wavfile as wav
from envelope import Envelope
from spectrogram import Spectrogram
from phonetic_data import PhoneData
from phonemes_and_features import PhoneInfo
from numpy.linalg import inv

class StimMatrix:

    def create_minmax_matrix(self, stimulus, sample_rate, first_sample):
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
        minimum = 100
        maximum = 400

        minimum = int((minimum/1000) * sample_rate)
        maximum = int((maximum/1000) * sample_rate)
        start = int(first_sample * sample_rate)

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

        print(matrix.shape)
        return matrix


    def create_single_lag_matrix(self, stimulus, sample_rate, first_sample):
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
        lag_point = 200
        lag_point = int((lag_point/1000) * sample_rate)
        lag = lag_point + 1
        start = int(first_sample * sample_rate)

        time = stimulus[0][start:].size
        matrix = np.zeros(shape=(lag, time))

        lag_step = 0

        count = 0
        while count < matrix.shape[0]:
            new_count = count + variables
            if count == 0:
                rows = stimulus[:, start:]
            else:
                rows = stimulus[:, start - lag_step:-lag_step]
            matrix[count: new_count] = rows
            lag_step += 1
            count = new_count

        return matrix

    def smoothing_matrix(self, dimension):
        """matrix created from the stimulus identity matrix. Meant to add bias to surrounding data points

        arguments and variables:
        dimension: the size of the matrix
        identity: the modified identity matrix
        indexes: tuples holding coordinates of the elements in the matrix with a value of 2
        arr_up: column locations one to the right of elements with a value of 2
        arr_down: column locations one to the left of elements with a value of 2

        returns:
        the modified smoothing matrix
        """

        identity = np.identity(dimension)
        identity[identity != 0] = 2
        identity[0][0], identity[-1][-1] = 1, 1
        identity[0][1], identity[-1][-2] = -1, -1
        indexes = np.where(identity == 2)
        arr_up = indexes[1] + 1
        arr_down = indexes[1] - 1
        identity[indexes[0], arr_up] = -1
        identity[indexes[0], arr_down] = -1

        return identity



    def get_stimulus(self, stimulus, requested_sample_rate, first_sample):
        """ builds and returns the stimulus matrix and the modified identity matrix
        """

        stim_matrix = self.create_minmax_matrix(stimulus, requested_sample_rate, first_sample)
        print('stim mat built')
        autocovariance = np.matmul(stim_matrix.transpose(), stim_matrix)
        print('autoco built')
        inverse = inv(autocovariance)
        print('inverse built')
        smoothing = self.smoothing_matrix(inverse.shape[0])
        print('smoothing built')

        return stim_matrix, inverse, smoothing

if __name__ == '__main__':

    trial = f'trials/sense_sentence_block_0'
    recording = f'{trial}.wav'
    sampling_rate, signal_data = wav.read(recording)
    time_array = np.arange(0, len(signal_data)) / sampling_rate

    envelope = Envelope()
    envelope.calculate_envelope(signal_data)

    stimulus_matrix = StimMatrix()
    envelope.downsample_envelope(sampling_rate)
    normalised_envelope = envelope.normalise_envelope()
    print(type(normalised_envelope))
    first_sample = 1.28
    normalised_envelope = np.asarray([normalised_envelope])

    # stim_matrix, inverse, smoothing = stimulus_matrix.get_stimulus(normalised_envelope, 1000, first_sample)
    stimul_matrix = stimulus_matrix.create_minmax_matrix(normalised_envelope, 1000, first_sample)
    # print(stim_matrix == stimul_matrix)

    # spectrogram = Spectrogram()
    # spectrogram.calculate_frequency_bands()
    # spectrogram.bandpass(signal_data, sampling_rate)
    # spectrogram.hilbert_transform()
    # spectrogram.downsample_spec(sampling_rate)
    #
    # sm = StimMatrix()
    # print(spectrogram.envelopes.shape)
    # sm.create_minmax_matrix(spectrogram.envelopes, 1000, first_sample)
    #¡¡
    # phonetic_data = PhoneData()
    # phonetic_data, features = phonetic_data.extract_json()
    #
    # phoneme_and_feature = PhoneInfo()
    #
    # time_array = np.arange(0, int(envelope.envelope.size)) / 1000
    # phoneme_and_feature.phoneme_information(f'{trial}.json')
    # phoneme_and_feature.calculate_var_array(time_array, phonetic_data, features)
    #
    # sm1 = StimMatrix()
    # sm1.create_minmax_matrix(phoneme_and_feature.phoneme_array, 1000, first_sample)
    # sm2 = StimMatrix()
    # sm2.create_minmax_matrix(phoneme_and_feature.features_array, 1000, first_sample)
    #


