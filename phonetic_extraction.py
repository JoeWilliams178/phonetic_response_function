import numpy as np
class PhoneExtract:

    def check_phoneme(self, phonetic_data, phoneme_input):

        p = next((phoneme for phoneme in phonetic_data if phoneme['phoneme'] == phoneme_input), None)
        if p is not None:
            return
        else:
            print(f'phoneme {phoneme_input} does not exist')
            exit()

    def get_phoneme_index(self, phonetic_data, phoneme_input):
        """cycles through all potential phoneme until it finds a match with the input

        arguments and variables:
        phonetic_data: all potential phonetic information collected from phoneme.json
        phoneme_input: phoneme present in trial
        phoneme['phoneme']: name of phoneme extracted from phonetic data

        returns the index of the input phoneme associated with its position in phonetic_data
        """

        for index, phoneme in enumerate(phonetic_data):
            if phoneme['phoneme'] == phoneme_input:
                return index


    def get_features_index(self,phonetic_data, features, phoneme_input):
        """recalls the phonetic features associated with a phoneme

        arguments and variables:
        phoneme: phoneme to be checked

        returns the corresponding phonetic features. manner, place and voice for consonants and backness for vowels
        """
        p = next((phoneme for phoneme in phonetic_data if phoneme["phoneme"] == phoneme_input), None)
        if p is not None and len(p) != 3:
            return [np.where(features == p["manner"])[0][0], np.where(features == p["place"])[0][0],
                    np.where(features == p["voicing"])[0][0]]
        elif p is not None:
            return np.where(features == p["backness"])[0][0]


if __name__ == '__main__':
    from phonetic_data import PhoneData
    pd = PhoneData()
    data, features = pd.extract_json()
    pe = PhoneExtract()
    pe.check_phoneme(data, "p")
    print(pe.get_features_index(data, features, "ah"))



