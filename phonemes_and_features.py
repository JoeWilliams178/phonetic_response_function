import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import json
from phonetic_data import PhoneData
from phonetic_extraction import PhoneExtract
from scipy import signal


class PhoneInfo:
    def __init__(self):
        """
        variables:
        phone_info = list containing phoneme information for each phoneme in file : start time, stop time, name
        phoneme_array = binary array for each phoneme. The array length will be equal to the number of sampling points
        features_array = binary array for each feature. The array length will be equal to the number of sampling points
        """
        self.phone_info = []
        self.phoneme_array = None
        self.features_array = None

    def phoneme_information(self, trial):
        """reads the json file containing the phonetic information associated with the trial
        """

        try:
            f = open(trial)
        except OSError:
            print(f'{trial} not found')
            exit()

        trial_data = json.load(f)
        f.close()

        for word in trial_data:
            self.collect_phones(word)

    def collect_phones(self, word):
        """reads through the phonemes in the input word extracting the start, end and name of each phoneme

        arguments and variables:
        word: a word extracted from the audio_signal
        no_of_phonemes: the number of phonemes in the word being assessed
        start: the starting time of the phoneme in seconds
        end: the end time in seconds of the phoneme
        phoneme: the name of the phoneme currently being assessed
        """
        if word['case'] != 'success':
            return
        no_of_phonemes = len(word['phones'])
        end = 0
        for index in range(0, no_of_phonemes):
            phoneme = word['phones'][index]
            if index == 0:
                start = word['start']
            else:
                start = end
            end = start + phoneme['duration']
            phoneme = phoneme['phone'][:2]
            if phoneme[1] == '_':
                phoneme = phoneme[:1]
            self.phone_info.append([start, end, phoneme])

    def calculate_var_array(self, time_array, phone_data, features):
        """reads through the phonemes in the input word extracting the start, end and name of each phoneme

            arguments and variables:
            word: a word extracted from the audio_signal
            no_of_phonemes: the number of phonemes in the word being assessed
            start: the starting time of the phoneme in seconds
            end: the end time in seconds of the phoneme
            phoneme: the name of the phoneme currently being assessed
            """

        phone_extract = PhoneExtract()


        phoneme_variable_arrays = np.zeros(shape=(len(phone_data), len(time_array)))
        features_variable_arrays = np.zeros(shape=(len(features), len(time_array)))

        for i, time in enumerate(time_array):
            for phoneme in self.phone_info:

                if phoneme[0] <= time < phoneme[1]:

                    index = phone_extract.get_phoneme_index(phone_data, phoneme[2])
                    phoneme_variable_arrays[index][i] = 1
                    feature_indices = phone_extract.get_features_index(phone_data, features, phoneme[2])
                    # print(type(feature_indices))
                    if type(feature_indices) is np.int64:
                        features_variable_arrays[feature_indices][i] = 1
                    else:
                        for feature_index in feature_indices:
                            # print(phoneme[2])
                            features_variable_arrays[feature_index][i] = 1
                    break

        self.no_overlap(phoneme_variable_arrays, time_array)
        self.phoneme_array = phoneme_variable_arrays
        self.features_array = features_variable_arrays

    @staticmethod
    def no_overlap(phoneme_variable_arrays, time_array):
        for index, time in enumerate(time_array):
            count = 0
            for array in phoneme_variable_arrays:
                if array[index] == 1:
                    count += 1
                    if count > 1:
                        print(f"""variables active at the same time. array: {phoneme_variable_arrays.index(array)}
                        , location: {index}""")
                        exit()

    def plot_multivar(self, time_array):
        for index, array in enumerate(self.phoneme_array):
            plt.plot(time_array, array, label=index)
            plt.legend(loc='right')
            plt.title("phonemes")
            plt.xlim(0, 2.5)
        plt.show()
        for index, array in enumerate(self.features_array):
            plt.plot(time_array, array, label=index)
            plt.legend(loc='right')
            plt.title("phonetic features")
        plt.show()

    def resample_phonemes_features(self, desired_freq, sampling_rate):
        temp_phonemes = []
        for phoneme in self.phoneme_array:
            length = int( phoneme.size * desired_freq / sampling_rate)
            resample = signal.resample(phoneme, length)
            new_phoneme = resample
            temp_phonemes.append(new_phoneme)

        self.phoneme_array = None
        self.phoneme_array = np.asarray(temp_phonemes)

        temp_features = []
        for feature in self.features_array:
            length = int(feature.size * desired_freq / sampling_rate)
            resample = signal.resample(feature, length)
            new_feature = resample
            temp_features.append(new_feature)

        self.features_array = None
        self.features_array = np.asarray(temp_features)

if __name__ == '__main__':
    phonetic_data = PhoneData()
    phonetic_data, features = phonetic_data.extract_json()

    low_trim = 1.28
    high_trim = 15.36
    trial = f'trials/sense_sentence_block_0'
    recording = f'{trial}.wav'
    sampling_rate, signal_data = wav.read(recording)
    signal_data = signal_data[int(low_trim * sampling_rate): int(high_trim * sampling_rate)]
    time_array = np.arange(0, len(signal_data)) / sampling_rate
    phoneme_and_feature = PhoneInfo()
    phoneme_and_feature.phoneme_information(f'trials/sense_sentence_block_0.json')
    phoneme_and_feature.calculate_var_array(time_array, phonetic_data, features)
    phoneme_and_feature.resample_phonemes_features(1000, sampling_rate)
    print(phoneme_and_feature.features_array.shape)

