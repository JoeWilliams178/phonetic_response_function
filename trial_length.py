import scipy.io.wavfile as wav
from envelope import Envelope
from spectrogram import Spectrogram
from phonetic_data import PhoneData
from phonemes_and_features import PhoneInfo
import numpy as np


conditions = ('sense', 'nonsense', 'sense_control', 'nonsense_control')
number_of_blocks = 29

phonetic_data = PhoneData()
phonetic_data, features = phonetic_data.extract_json()

for cnd in conditions:
    for block in range(1, number_of_blocks + 1):
        trial = f'trials/{cnd}_sentence_block_{block}'
        recording = f'{trial}.wav'
        sampling_rate, signal_data = wav.read(recording)

        envelope = Envelope()
        envelope.calculate_envelope(signal_data)
        envelope.downsample_envelope(sampling_rate)

        spectrogram = Spectrogram()
        spectrogram.calculate_frequency_bands()
        spectrogram.bandpass(signal_data, sampling_rate)
        spectrogram.hilbert_transform()
        spectrogram.downsample_spec(sampling_rate)

        phoneme_and_feature = PhoneInfo()
        time_array = np.arange(0, int(envelope.envelope.size)) / 1000
        phoneme_and_feature.phoneme_information(f'{trial}.json')
        phoneme_and_feature.calculate_var_array(time_array, phonetic_data, features)



        if cnd == 'sense' and block == 1:
            old_length = len(signal_data)
            old_env_length = len(envelope.envelope)
            old_spec_length = len (spectrogram.envelopes[0])
            old_phone_length = len(phoneme_and_feature.phoneme_array[0])
            old_feature_length = len(phoneme_and_feature.features_array[0])
        else:
            old_length = signal_length
            old_env_length = env_length
            old_spec_length = spec_length
            old_phone_length = phone_length
            old_feature_length = feature_length

        signal_length = len(signal_data)
        env_length = len(envelope.envelope)
        spec_length = len(spectrogram.envelopes[0])
        phone_length = len(phoneme_and_feature.phoneme_array[0])
        feature_length = len(phoneme_and_feature.features_array[0])

        if signal_length != old_length:
            print(f'{cnd} trial {block} is a different length')
        if env_length != old_env_length:
            print(f'{cnd} trial {block}  envelope is a different length')
        if spec_length != old_spec_length:
            print(f'{cnd} trial {block}  spectrograms are a different length')
        if phone_length != old_phone_length:
            print(f'{cnd} trial {block}  phonemes are a different length')
        if feature_length != old_feature_length:
            print(f'{cnd} trial {block}  features are a different length')
        if feature_length != phone_length != spec_length != env_length:
            print(f'variables in {cnd} trial {block} do not match length')





