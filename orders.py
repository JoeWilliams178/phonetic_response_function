def butter_bandpass(lower_limit, upper_limit, fs, order=5):
    nyq = 0.5 * fs
    low = lower_limit / nyq
    high = upper_limit / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def round_value(frequency):
    if frequency < 1000:
        multiplier = 0.1
    else:
        multiplier = 0.01
    return math.ceil(frequency * multiplier) / multiplier


def check_order(lower_limit, upper_limit, sample_rate):
    """ establish the optimal order for the bandpass filter

    arguments and variables:
    lower_limit: lower end value of the frequency band
    upper_limit = upper end value of the frequency band
    sample_rate = sample_rate of the audio input
    kernel_b = b coefficient for weighted values of the original data points
    kernel_a = a coefficient for weighted values from the already filtered signal
    """
    for order in [2, 3, 4, 5, 6, 7]:
        kernel_b, kernel_a = butter_bandpass(lower_limit, upper_limit, sample_rate, order=order)
        w, h = signal.freqz(kernel_b, kernel_a, worN=2000)
        plt.plot((sample_rate * 0.5 / np.pi) * w, abs(h), label='order = %d' % order)

    plt.plot([0, lower_limit, lower_limit, upper_limit, upper_limit, sample_rate / 2], [0, 0, 1, 1, 0, 0], 'r')
    plt.xlabel('Frequency (Hz)')
    plt.xlim(lower_limit - 500, upper_limit + 1000)

    plt.show()


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    import scipy.io.wavfile as wav
    import math

    # conditions = ('sense', 'nonsense', 'sense_control', 'nonsense_control')
    # for cnd in conditions:
    #     number_of_blocks = 29
    #     for block in range(1, number_of_blocks + 1):
    #         trial = f'trials/{cnd}_sentence_block_{block}'
    #         recording = f'{trial}.wav'
    #         print(f'{trial}.wav')
    #         sampling_rate, data = wav.read(recording)
    #         check_order()

    trial = f'trials/sense_sentence_block_1'
    recording = f'{trial}.wav'
    print(f'{trial}.wav')
    sample_rate, data = wav.read(recording)

    factor = math.pow(2, 5. / 16)
    frequency_limit = 250
    rounded_limit = 250

    for i in range(1, 17):
        rounded_limit_previous = rounded_limit
        frequency_limit *= factor
        rounded_limit = int(round_value(frequency_limit))
        if rounded_limit == 8000:
            rounded_limit = 7999
        check_order(rounded_limit_previous, rounded_limit, sample_rate)
