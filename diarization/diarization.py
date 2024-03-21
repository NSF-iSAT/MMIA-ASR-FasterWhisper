from speaker_diarization_chunk import SpeakerDiarization
from scipy.io import wavfile
import json
import os


def diarization_init(refSpeakers):
    sd = SpeakerDiarization(refSpeakers=refSpeakers, vad=True, compositional=False)
    return sd


def diarization_predict(sd, input_file, output_file):
    sampleRate, data = wavfile.read(input_file)
    f = open(output_file, "w")
    f.close()
    n_chunks = (int(len(data)/16000))//10 + int(int(len(data)/16000)%10>0)
    for i in range(n_chunks):
        # replace relative time stamps with global time stamps
        results = sd.getResults(data[(i*10)*sampleRate:(i*10+10)*sampleRate])
        for key in list(results.keys()):
            results[str(float(key) + (i*10.0))] = 'speaker:'+str(results[key][0])
            if i !=0 :
                del results[key]

        # read what is in json and add new contents
        with open(output_file, 'r') as j:
            if not os.stat(output_file).st_size == 0:
                contents = json.loads(j.read())
                contents[f'segmentID: {input_file}'].update({f'chunk{i}':results})
            else:
                contents = dict()
                contents[f'segmentID: {input_file}'] = {f'chunk{i}':results}
        # write latest content to the json
        with open(output_file, 'w+') as outfile:
            outfile.write(json.dumps(contents))



#sd = diarization_init(refSpeakers = ['example_data/spk0.wav', 'example_data/spk1.wav', 'example_data/spk2.wav'])
#diarization_predict(sd, 'example_data/example.wav', output_file="json_data.json")

