import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt


class Envelope:

    def __init__(self):
        self.envelope = None

    def calculate_envelope(self, audio_signal):
        """Calculate the envelope of the provided signal data

        arguments and variables:
        audio_signal: input audio signal
        analytic_signal = the computed hilbert transform of data
        envelope = the absolute of the hilbert transform

        if no hilbert transform computed throws type error as broadband_envelope remains None
        """

        analytic_signal = signal.hilbert(audio_signal)
        self.envelope = np.abs(analytic_signal)

        if self.envelope is None:
            raise TypeError('Failed to calculate broadband envelope')

    def plot_envelope(self, audio_signal, time):
        """plot the signal envelope

        arguments and variables:
        audio_signal: input audio signal
        time = time array corresponding to data
        """

        plt.plot(time, self.envelope, 'r', label='signal envelope')
        # plt.plot(time, audio_signal, label='raw signal')
        plt.ylabel('Amplitude')
        plt.xlabel('time (s)')
        plt.legend()
        plt.show()

    def get_envelope(self):
        """returns the signal envelope
        """

        return self.envelope

    def normalise_envelope(self):
        minimum = np.amin(self.envelope)
        maximum = np.amax(self.envelope)
        normalised_envelope = np.array(self.envelope, copy=True)

        for index, point in enumerate(normalised_envelope):
            x = (point-minimum) / (maximum-minimum)
            normalised_envelope[index] = x
        return normalised_envelope

    def downsample_envelope(self, old_freq):
        new_freq = 1000
        length = int(self.envelope.size * new_freq / old_freq)
        resample = signal.resample(self.envelope, length)
        self.envelope = resample

if __name__ == '__main__':
    trial = f'trials/sense_sentence_block_1'
    recording = f'{trial}.wav'
    sampling_rate, signal_data = wav.read(recording)
    time_array = np.arange(0, len(signal_data)) / sampling_rate
    print(len(time_array))
    envelope = Envelope()
    envelope.calculate_envelope(signal_data)
    print(envelope.normalise_envelope().size)
    envelope.downsample_envelope(sampling_rate)
