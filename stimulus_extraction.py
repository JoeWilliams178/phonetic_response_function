import requests
import json

conditions = ['sense', 'sense_control', 'nonsense', 'nonsense_control']

params = (
    ('async', 'false'),
)
for condition in conditions:
    for trial in range(0, 30):
        audio = condition + '_sentence_block_' + str(trial) + '.wav'
        transcript = condition + '_sentence_block_' + str(trial) + '.txt'
        json_words = condition + '_sentence_block_' + str(trial) + '.json'
        print(audio)
        files = {
            'audio': (audio, open('trials/' + audio, 'rb')),
            'transcript': (transcript, open('trials/' + transcript, 'rb')),
        }
        response = requests.post('http://localhost:8765/transcriptions', params=params, files=files)
        text = response.text
        output = json.loads(text)
        words = output['words']
        with open('trials/' + json_words, 'w') as fp:
            json.dump(words, fp)
