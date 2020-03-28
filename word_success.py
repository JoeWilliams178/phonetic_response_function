import json

def check_word(words, trial):
    for number, word in enumerate(words):
        if number == 51:
            print(word['end'])
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
        check_word(data, trial)
        if len(data) != 52:
            print(f'fail{trial}')
