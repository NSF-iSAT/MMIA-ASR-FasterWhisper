# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
from ctypes import *
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    return
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so')
asound.snd_lib_error_set_handler(c_error_handler)

# Import required libraries
from faster_whisper import WhisperModel
import pyaudio
import wave
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
from itertools import combinations
from collections import OrderedDict
from pydub import AudioSegment
#from pyannote.audio import Task, TaskOutput, TaskType
from pyannote.audio.train.task import Task, TaskOutput, TaskType
from pyannote.audio.models import SincTDNN
from diarization.voice_activity_detection import voice_activity_detection

# Make Torch use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
print("Running on:",device)

def record_chunk(p, stream, file_path, chunk_length=3):
    """
    Function that retrives that recorded audio of length "chunk_length", saves it in file_path and returns the audio in numpy format.
    """
    frames=[]
    for _ in range(0, int(16000/1024*chunk_length)):
        data= stream.read(1024)
        frames.append(data)
    
    # Save the audio in the mentioned file_path
    wf= wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Return the audio in numpy format
    final = b''.join(frames)
    audio_np = np.frombuffer(final, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

def extract_audio_between_timestamps(audio_np, sample_rate, start_ts, end_ts):
    """
    Function to extract audio from audio_np between start_ts and end_ts timestamps.
    
    Parameters:
        audio_np (numpy.ndarray): NumPy array containing the audio data.
        sample_rate (int): Sampling rate of the audio data.
        start_ts (float): Start timestamp (in seconds).
        end_ts (float): End timestamp (in seconds).
    
    Returns:
        numpy.ndarray: Extracted audio data between start_ts and end_ts timestamps.
    """
    # Convert timestamps to indices
    start_idx = int(start_ts * sample_rate)
    end_idx = int(end_ts * sample_rate)
    
    # Slice the audio_np array to extract the desired portion
    extracted_audio = audio_np[start_idx:end_idx]
    
    return extracted_audio

def AudioSegment_to_np_array(asg):
    """
    Funtion to convert audio files to np array
    """
    dtype = getattr(np, "int{:d}".format(asg.sample_width * 8))  # Or could create a mapping: {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}
    arr = np.ndarray((int(asg.frame_count()), asg.channels), buffer=asg.raw_data, dtype=dtype)
    return arr

def load_video_to_ndarray(filepath):
    """load video and convert to numpy 1-d ndarray of PCM data representing audio signal
    Args:
        filepath (str): path to media file (must be an audio or video file supported by ffmpeg)

    Returns:
        sig: _description_
        sig_nsamp
    """
    #load chunk and covert to 16k mono pcm on the fly
    chunk_audio = AudioSegment.from_file(filepath)
    SAMPLE_RATE = 16000  # Hz, see load_video for requirements
    chunk_audio = chunk_audio.set_channels(1).set_sample_width(2).set_frame_rate(SAMPLE_RATE)
    sig = AudioSegment_to_np_array(chunk_audio).flatten()
    sig_nsamp = len(sig)
    return sig, sig_nsamp

class gNet (nn.Module):
    def __init__ (self, dim):
        super(gNet, self).__init__()
        self.linear1a = nn.Linear(dim, dim)
        self.linear1b = nn.Linear(dim, dim)

    def forward (self, X1, X2):
        linear = self.linear1a(X1) + self.linear1a(X2) + self.linear1b(X1*X2)
        return linear
    
def pairwiseDists (A, B):
    A = A.reshape(A.shape[0], 1, A.shape[1])
    B = B.reshape(B.shape[0], 1, B.shape[1])
    D = A - B.transpose(0,1)
    #print("----------------------------------------------------------------------")
    #print(A.shape)
    #print(B.shape)
    #print(D.shape)
    #print("Row Sum:", torch.sum(D, dim=2))
    #print("Col Sum:", torch.sum(D, dim=1))
    #print("Col Sum:", torch.norm(D, p=2, dim=2))
    #print("----------------------------------------------------------------------")
    return torch.norm(D, p=2, dim=2)

def pairwiseDists_WeighetedVote (A, B):
    A = A.reshape(A.shape[0], 1, A.shape[1])
    B = B.reshape(B.shape[0], 1, B.shape[1])
    D = A - B.transpose(0,1)
    print("Old Distance:", torch.norm(D, p=2, dim=2))
    return torch.sum(D, dim=2)

def cosine_similarity(A, B):
    dot_product = torch.dot(A.flatten(), B.flatten())
    norm_A = torch.norm(A)
    norm_B = torch.norm(B)
    similarity = dot_product / (norm_A * norm_B)
    return similarity

class SpeakerDiarization:
    def getData(self, audio):
        if isinstance(audio, str):
            sampleRate, data = wavfile.read(audio)
            assert sampleRate == 16000, "The sample rate of audio should be 16000"
            return data
        else:
            data, rate = audio
            return data

    def getEmb(self, data, mean=False, padding=False):
        if padding:
            data = np.pad(data, (self.sampleRate, self.sampleRate), mode='constant', constant_values=0)
        datatensor = []
        for i in range(int(data.shape[0]/self.sampleRate/self.stepSize - 2/self.stepSize)):
            datatensor.append(data[int(i*self.stepSize*self.sampleRate):int(i*self.stepSize*self.sampleRate) + 2*self.sampleRate])
        datatensor = np.stack(datatensor)
        datatensor = torch.from_numpy(datatensor).float().to(device)
        with torch.no_grad():
            embs = self.fNet(datatensor.unsqueeze(-1))
        if mean:
            embs = torch.mean(embs, 0)
        return embs

    def __init__(self, refSpeakers, checkpointPath=os.path.join('diarization', 'checkpoints'),
                 vad=False, compositional=False, stepSize=0.1):
        '''
        refSpeakers: a dict where keys are speakerID (str) and values represent enrollment audio files, one per speakerID.
            dict values can be the path of a .wav file OR a 1-d numpy ndarray of the waveform.
            The audio should be 1 channel with sample rate of 16000
        checkpointPath: path of saved models
        vad: whether to use vad in diarization, if True, google WebRTC will be used
        compostional: whether to enable multi-speaker support
        '''
        self.sampleRate = 16000
        self.stepSize = stepSize

        # Create and load diarization model
        task = Task(TaskType.REPRESENTATION_LEARNING,TaskOutput.VECTOR)
        specifications = {'X':{'dimension': 1} ,'task': task}
        sincnet = {'instance_normalize': True, 'stride': [5, 1, 1], 'waveform_normalize': True}
        tdnn = {'embedding_dim': 512}
        embedding = {'batch_normalize': False, 'unit_normalize': False}
        self.fNet = SincTDNN(specifications=specifications, sincnet=sincnet, tdnn=tdnn, embedding=embedding).to(device)
        if not compositional:
            self.fNet.load_state_dict(torch.load(os.path.join(checkpointPath, 'f_vxc.pt'), map_location=device))
        else:
            self.fNet.load_state_dict(torch.load(os.path.join(checkpointPath, 'best_f.pt'), map_location=device))
            self.gNet = gNet(512).to(device)
            self.gNet.load_state_dict(torch.load(os.path.join(checkpointPath, 'best_g.pt'), map_location=device))
            self.gNet.eval()
        self.fNet.eval()

        self.vad = vad
        self.compositional = compositional

        # Create cluster centroids based on reference speaker samples
        self.clusters = []
        self.label = {}
        for cnt,(speakerID, audio) in enumerate(refSpeakers.items()):
            self.label[cnt] = (speakerID,)
            refData = self.getData(audio)
            refEmbs = self.getEmb(refData, mean=True)
            self.clusters.append(refEmbs)
        self.clusters = torch.stack(self.clusters)

        if compositional:
            combsList = list(combinations(list(refSpeakers.keys()), 2))
            for cCnt, c in enumerate(combsList):
                self.label[len(refSpeakers)+cCnt] = c
            combs = torch.tensor(range(len(combsList)))
            comb2A = self.clusters[combs.transpose(-2, -1)[0]]
            comb2B = self.clusters[combs.transpose(-2, -1)[1]]
            merged = self.gNet(comb2A, comb2B)
            self.clusters = torch.cat([self.clusters, merged], 0)

        self.clustersNorm = F.normalize(self.clusters)

    def getResults(self, audio):
        '''
        returns a dictionary whose key is timestamp and value is a tuple of speakerID.
        '''
        #print("audio type, length and width:", type(audio), len(audio), len(audio[0]))
        data = self.getData((audio, len(audio)))
        print("data type, length and width:", type(data), len(data), len(data[0]))
        #print("Audio Length:", len(audio))
        if(len(audio) == 0):
            return OrderedDict()
        embs = self.getEmb(data, padding=True) # Test Size: 4 x 512
        embsNorm = F.normalize(embs)
        print("embsNorm type, length and width:", type(embsNorm), len(embsNorm), len(embsNorm[0]))
        print(embsNorm[0:5])
        #print("Embedding type, length and width:", type(embsNorm), len(embsNorm), len(embsNorm[0]))
        #print("self.clustersNorm enrollment embediing type, length and width:", type(self.clustersNorm), len(self.clustersNorm), len(self.clustersNorm[0]))
        #dists = cosine_similarity(embsNorm, self.clustersNorm) # Find the cosine distance for the new audio to every cluster center. Length will be number of clusters.
        dists = pairwiseDists(embsNorm, self.clustersNorm) # Test Size: 4 x 3 # Find the eucledian distance for the new audio to every cluster center. Length will be number of clusters.
        #dists = pairwiseDists_WeighetedVote(embsNorm, self.clustersNorm) # Find the cosine distance for the new audio to every cluster center. Length will be number of clusters.  
        print("dists type, length and width:", type(dists), len(dists), len(dists[0]))
        print("dists printing:\n",dists)
        preds = torch.argmin(dists, 1) # Find the index with lowest distance. Length will be just 1.
        print("preds printing:\n",preds)
        #print("preds type, length and width:", type(preds), len(preds), len(preds[0]))

        # get vad results if vad flag is true
        if self.vad:
            vadRes = np.array(voice_activity_detection(data, self.sampleRate, 1)).astype(int)
            numSegments = int(self.stepSize*1000/20)
            padLength = numSegments - vadRes.shape[0]%numSegments
            vadRes = np.pad(vadRes, (0, padLength), mode='symmetric')
            vadRes = np.reshape(vadRes, (-1, numSegments))
            vadRes = np.sum(vadRes, 1) > 2

        result = OrderedDict()
        for pCnt, p in enumerate(preds): # pCnt will be zero always since we are just calculating 1 speaker in the entire audio.
            print("pCnt, p:",pCnt, p)
            timestamp = pCnt * self.stepSize
            label = self.label[p.item()]
            if self.vad:
                if vadRes[pCnt]:
                    result['{:0.1f}'.format(timestamp)] = label
            else:
                result['{:0.1f}'.format(timestamp)] = label
            print("---------------------Result within---------------------")
            print(result)

        return result

def main():

    # Initialize the fast whisper models
    model_size= "large-v3" #"medium" "distil-large-v2"
    model= WhisperModel(model_size, device= str(device), compute_type= "float16")

    # Start recording the audio
    sample_rate= 16000
    p= pyaudio.PyAudio()
    stream= p.open(format= pyaudio.paInt16, channels=1, rate= sample_rate, input=True, frames_per_buffer= 1024)
    print("Recording Started...")

    accumulated_transcripts= ""

    # Initialze the diarizer with recorded enrollments
    enrollment_utterances = {'Niranjan':load_video_to_ndarray('Enrollment_Niranjan.wav'),
                             'Kamala Harris':load_video_to_ndarray('Enrollment_Kamala_Harris.wav'), 
                             'Trump':load_video_to_ndarray('Enrollment_Trump.wav')}
    diarizer = SpeakerDiarization(refSpeakers=enrollment_utterances, vad=False, compositional=False, stepSize=2) #stepSize decides the frequency of diarization. Say if it is 0.5 then the result will be given as a speaker ID for every 0.5secs of the input audio in a dictionary.

    try:
        while True:
            # Retrive the recorded audio and transcribe them
            chunk_length=3 # Define the frequency of recording in seconds
            chunk_file= 'temp_chunk.wav'
            recorded_audio= record_chunk(p, stream, chunk_file, chunk_length=chunk_length)
            segments, info= model.transcribe(chunk_file, beam_size= 5)
            os.remove(chunk_file)

            # Unpack the segments and diarize them individually. 
            for segment in segments:
                #print("Segment entering")
                start_time = segment.start # Start time of the segment
                end_time = segment.end # End time of the segment
                text = segment.text  # Transcribed text of the segment
                accumulated_transcripts+= text+" "
                #print(start_time, "to", end_time)
                
                # Calculate the closest matching speaker in the segment
                audio_to_diarize=  extract_audio_between_timestamps(recorded_audio, sample_rate, start_time, end_time) # Extract the audio using timeframe
                #print(len(audio_to_diarize))
                try:
                    diarizer_result = diarizer.getResults(audio_to_diarize)
                    print("--------------------- diar result -------------------------------")
                    print(diarizer_result.values())
                    
                    speaker= [i for i in diarizer_result.values()][0][0]
                    print("--------------------- speaker 1 -------------------------------")
                    print(speaker)
                    speaker= diarizer_result['0.0'][0] # This is just taking the first detected speaker, rather just the mode of speakers.
                    print("--------------------- speaker 2 -------------------------------")
                    #print("Speaker:",speaker)
                    print(speaker,":", text) 
                except:
                    continue

                """
                #print("audio_to_diarize completed")
                # Save the audio in the mentioned file_path
                File_path= "segment_check_"+str(k)+".wav"
                k+=1
                wf= wave.open(File_path, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(2)
                #wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(sample_rate)
                wf.writeframes(audio_to_diarize.tobytes())
                wf.close()
                #print("Completed writing")
                
                """

                #print(speaker,":", text)           
    except KeyboardInterrupt:
        print("Stopping...")

        with open("log.text", 'w') as log_file:
            log_file.write(accumulated_transcripts)

    finally:
        print("LOG:", accumulated_transcripts)
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()



