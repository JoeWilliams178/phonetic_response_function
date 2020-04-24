# Separates an audio signal into different frequency bands at biologically relevant spacing intervals.
# These frequency ranges are calculated by applying a bandpass filter (butterworth). See orders.py to explore the
# effect of using different orders in the bandpass filter. The hilbert transformation is applied to each signal and
# the resultant values are plotted with time to produce the spectrogram.

import math
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack as fft
from envelope import Envelope
from scipy import stats
import os


class Spectrogram:
    def __init__(self):
        self.bands = []
        self.bandpass_signals = []
        self.envelopes = None

    @staticmethod
    def round_value(frequency):
        """rounds frequency values.  values < 1000 rounded to the 10 else 100.

        arguments and variables:
        frequency: frequency value to be rounded
        multiplier: specifies rounding to the nearest 10 or 1000

        returns the rounded value
        """
        if frequency < 1000:
            multiplier = 0.1
        else:
            multiplier = 0.01
        return math.ceil(frequency * multiplier) / multiplier

    def calculate_frequency_bands(self):
        """establishes the appropriate frequency bands ranging from 250 - 8000Hz. resultant bands stored in a list

        variables:
        factor: multiple to get appropriate band range
        frequency_limit = upper or lower band value
        rounded_limit: band value rounded to nearest 10 or 100
        frequency_limits: band values
        """

        factor = math.pow(2, 5. / 16)

        frequency_limit = 250
        frequency_limits = [frequency_limit]

        for i in range(1, 17):
            frequency_limit *= factor
            rounded_limit = int(self.round_value(frequency_limit))
            if rounded_limit == 8000:
                rounded_limit = 7999
            frequency_limits.extend([rounded_limit, rounded_limit])

        it = iter(frequency_limits)
        self.bands = list(zip(it, it))

    def bandpass(self, audio_signal, sample_rate):
        """bandpass at specified frequency ranges. Achieved using the butterworth filter

        arguments and variables:
        audio_signal: signal data
        sr: sample rate of the signal
        nyquist: nyquist value corresponding to the signal
        low and high: upper and lower limits of the band
        kernel_b = b coefficient for weighted values of the original data points
        kernel_a = a coefficient for weighted values from the already filtered signal
        """

        nyquist = sample_rate / 2

        for band in self.bands:
            low = band[0] / nyquist
            high = band[1] / nyquist
            order = 4
            kernel_b, kernel_a = signal.butter(order, np.array([low, high]), btype="bandpass")
            bandpass_signal = signal.lfilter(kernel_b, kernel_a, audio_signal)
            self.bandpass_signals.append(bandpass_signal)

    def filtered_signals_plot(self, time, sr):
        """plots the filtered signal information at each frequency band.

        arguments and variables:
        time: array of time points corresponding to the signal data
        sr: sample rate of the signal
        """

        for i in range(len(self.bandpass_signals)):
            plt.title(f'Frequency Range: {self.bands[i]}')
            plt.plot(time, self.bandpass_signals[i])
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.show()

            data_points = len(self.bandpass_signals[i])
            frequency_range = np.linspace(0, sr / 2, int(np.floor(data_points) / 2) + 1)
            power = np.abs(fft.fft(signal.detrend(self.bandpass_signals[i])) / data_points) ** 2
            plt.plot(frequency_range, power[0:len(frequency_range)])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.title(f'Butterworth filter: power spectrum {self.bands[i]}')
            plt.ylim(0, 0.1)
            plt.xlim([0, 8000])
            plt.show()

    def hilbert_transform(self):
        """applies the hilbert transform to each filtered signals. Essentially establishing the envelope at each
        frequency range

        arguments and variables:
        e: envelope class object
        self.envelopes: array containing the calculated envelopes at each frequency band
        """
        envelopes = []

        for i in range(len(self.bandpass_signals)):
            e = Envelope()
            e.calculate_envelope(self.bandpass_signals[i])
            envelope = e.envelope
            envelopes.append(envelope)

        self.envelopes = np.asarray(envelopes)

    def normalise_spec(self):
        """normalises each envelope in the spectrogram between 0 and 1. It does not change the values associated with
        the spectrogram class object

        variables:
        normalised envelopes: spectrogram containing the normalised envelopes at each frequency band
        minimum: minimum value found in the envelope
        maximum: maximum value found in the envelope

        retruns the spectrogram as an np.array of normalised envelopes
                """

        normalised_envelopes = []
        # for ind, envelope in enumerate(self.envelopes):
        #
        #     minimum = np.amin(envelope)
        #     maximum = np.amax(envelope)
        #     normalised_envelope = np.array(envelope, copy=True)
        #
        #     for index, point in enumerate(normalised_envelope):
        #         x = (point - minimum) / (maximum - minimum)
        #         normalised_envelope[index] = x
        #     normalised_envelopes.append(normalised_envelope)
        for envelope in self.envelopes:
            normalised_envelope = stats.zscore(envelope)
            normalised_envelopes.append(normalised_envelope)

        return np.asarray(normalised_envelopes)

    def plot_envelopes(self, time):
        """plots the envelopes at each frequency range produced by the hilbert transform
        """

        for i, envelope in enumerate(self.envelopes):
            plt.plot(time, envelope, 'r', label='broadband amplitude envelope')
            plt.plot(time, self.bandpass_signals[i], label='raw signal')
            plt.ylabel('Amplitude')
            plt.xlabel('time (s)')
            plt.legend()
            plt.show()

    def plot_spectrogram(self, time):
        """decimate is used to downsample the signals. Plots the heat map which represents the spectrogram
        """
        plot = plt.imshow(self.envelopes, cmap='hot', interpolation='nearest', extent=[0, time[-1], 15, 0])
        ax = plot.axes
        ax.invert_yaxis()
        plt.xlabel('time (s)')
        plt.colorbar(plot)
        plt.show()

    def downsample_spec(self, old_freq, new_freq):
        """downsamples all envelopes in the spectrogram to a new sample rate
        """

        temp_envs = []
        for envelope in self.envelopes:
            length = int(envelope.size * new_freq / old_freq)
            resample = signal.resample(envelope, length)
            envelope = resample
            temp_envs.append(envelope)

        self.envelopes = np.asarray(temp_envs)



if __name__ == '__main__':
    trial = f'trials/sense_sentence_block_1'
    recording = f'{trial}.wav'
    sampling_rate, signal_data = wav.read(recording)
    time_array = np.arange(0, len(signal_data)) / sampling_rate
    spectrogram = Spectrogram()
    spectrogram.calculate_frequency_bands()
    spectrogram.bandpass(signal_data, sampling_rate)
    # spectrogram.filtered_signals_plot(time_array, sampling_rate)
    spectrogram.hilbert_transform()
    # spectrogram.plot_envelopes(time_array)
    spectrogram.plot_spectrogram(time_array)
    print(spectrogram.envelopes[0].size)
    spectrogram.downsample_spec(sampling_rate, 1000)
    print(spectrogram.envelopes[0].size)
    time_array = np.arange(0, spectrogram.envelopes[0].size) / 1000
    spectrogram.plot_spectrogram(time_array)
    print(max(spectrogram.normalise_spec()[0]))
    print(min(spectrogram.normalise_spec()[0]))
    print(max(spectrogram.envelopes[0]))
