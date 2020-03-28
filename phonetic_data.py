import json
import numpy as np


class PhoneData:

    def extract_json(self):
        """reads phoneme.json which contains information about all phonemes and phonetic features.

        creates:
        data: contains all phonetic information
        """
        try:
            file = 'phoneme.json'
            f = open(file)
        except OSError:
            print(f'{file} not found')
            exit()

        data = json.load(f)
        f.close()
        data = data['phonemes']

        features = []
        for phoneme in data['consonants']:
            features.extend([phoneme['manner'], phoneme['place'], phoneme['voicing']])

        for phoneme in data['vowels']:
            features.append(phoneme['backness'])

        features = np.unique(features)

        return data['consonants'] + data['vowels'], features


if __name__ == '__main__':
    pd = PhoneData()
    data, features = pd.extract_json()