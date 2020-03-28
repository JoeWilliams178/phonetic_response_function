import json


class Features:
    def __init__(self):
        """reads phoneme.json which contains information about all phonemes and phonetic features.

        variables:
        data: contains all phonetic information
        phonemes: contains the names of phonemes only, in an ordered list used for indexing
        """

        try:
            file = 'phoneme.json'
            f = open(file)
        except OSError:
            print(f'{file} not found')
            exit()

        data = json.load(f)
        f.close()
        self.data = data['phonemes']['consonants'] + data['phonemes']['vowels']
        self.features = set()
        for phoneme in self.data:
            if len(phoneme) == 5:
                self.features.update([phoneme['manner'], phoneme['place'], phoneme['voicing']])
            else:
                self.features.add(phoneme['backness'])
        self.features = list(self.features)

    def check_phoneme(self, phoneme_input):

        p = next((phoneme for phoneme in self.data if phoneme['phoneme'] == phoneme_input), None)

        if p is None:
            print(f'phoneme {phoneme_input} does not exist')
            exit()

    def get_phoneme_index(self, phoneme_input):
        phoneme = None
        index = 0
        while phoneme != phoneme_input:
            phoneme = self.data[index]['phoneme']
            index += 1

        return index - 1

    def get_phoneme(self, index_value):
        return

    def get_features(self, phoneme_input):
        """recalls the phonetic features associated with a phoneme

        arguments:
        phoneme: phoneme to be checked

        returns the corresponding phonetic features. manner, place and voice for consonants and backness for vowels
        """
        p = next((phoneme for phoneme in self.data['consonants'] if phoneme['phoneme'] == phoneme_input), None)
        if p is not None:
            return [self.features.index(p['manner']), self.features.index(p['place']),
                    self.features.index(p['voicing'])]
        else:
            p = next((phoneme for phoneme in self.data['vowels'] if phoneme['phoneme'] == phoneme_input), None)
            if p is not None:
                return self.features.index(p['backness'])


if __name__ == '__main__':
    feature = Features()
    feature.check_phoneme("p")
    print(feature.features)
