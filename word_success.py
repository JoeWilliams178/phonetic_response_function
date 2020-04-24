import json


def check_last_word_success(word, trial):
    """checks whether aligner successfully aligned the final word in the trial

    arguments:
    word: final word aligned
    trial: trial information
    """

    if word['case'] != 'success':
        print(word, trial)

def check_last_word_length(word, trial):
    """checks whether aligner successfully aligned words contain all phonetic information. requires visual inspection

    arguments:
    word: final word aligned
    trial: trial information
    """

    if word['case'] == 'success' and len(word['phones']) < 3:
        print(word, trial)

def check_word(words, trial):
    """checks whether aligner successfully aligned all words in the trial

    arguments:
    words: all words aligned
    trial: trial information
    """

    for number, word in enumerate(words):
        if word['case'] == 'success' and len(word['phones']) < 3:
            print(word['word'], ' ', word['phones'], number)
        if word['case'] != 'success':
            print(word, '\n', trial)


conditions = ('sense', 'nonsense', 'sense_control', 'nonsense_control')
number_of_blocks = 29
for cnd in conditions:
    for block in range(1, number_of_blocks + 1):
        trial = f'trials/{cnd}_sentence_block_{block}.json'
        try:
            f = open(trial)
        except OSError:
            print(f'{trial} not found')
            exit()
        data = json.load(f)
        f.close()
        check_last_word_success(data[51], trial)
        check_last_word_length(data[51], trial)
        check_word(data[:48], trial)
        if len(data) != 52:
            print(f'fail{trial}')
