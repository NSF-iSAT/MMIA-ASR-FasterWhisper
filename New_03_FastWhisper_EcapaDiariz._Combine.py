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

from New_02_ECAPA_Diarization import pairwiseDists
from New_02_ECAPA_Diarization import SpeakerDiarizationChunkEcapa as SpeakerDiarization
from statistics import mode

from anonymization.anonymization import anonymize_text, anonymize_text_with_deny_list
import json
from decimal import Decimal


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

def main():

    # Initialize the fast whisper models
    model_size= "large-v3" #"medium" "distil-large-v2"
    model= WhisperModel(model_size, device= str(device), compute_type= "float16")

    # Start recording the audio
    SAMPLE_RATE = 16000  # Hz, see load_video for requirements
    CHUNK_LENGTH = 3  # Define the frequency of recording in seconds
    MIN_CHUNK_LEN = 0.1 # seconds, chunks shorter than this will be skipped
    p= pyaudio.PyAudio()
    stream= p.open(format= pyaudio.paInt16, channels=1, rate= SAMPLE_RATE, input=True, frames_per_buffer= 1024)
    print("Recording Started...")

    # Variable the transcript to collect all the transcripts in this session for logging
    accumulated_transcripts= ""

    # Initialze the diarizer with recorded enrollments
    enrollment_utterances = {'Niranjan':load_video_to_ndarray('Enrollment_Niranjan.wav'),
                             'Kamala Harris':load_video_to_ndarray('Enrollment_Kamala_Harris.wav'), 
                             'Trump':load_video_to_ndarray('Enrollment_Trump.wav')}
    diarizer = SpeakerDiarization(refSpeakers=enrollment_utterances) #stepSize decides the frequency of diarization. Say if it is 0.5 then the result will be given as a speaker ID for every 0.5secs of the input audio in a dictionary.
    sessionID= 1
    chunk_number= 1

    try:
        while True:
            # Retrive the recorded audio and transcribe them
            chunk_file= 'temp_chunk.wav'
            recorded_audio= record_chunk(p, stream, chunk_file, chunk_length=CHUNK_LENGTH)
            segments, info= model.transcribe(chunk_file, beam_size= 5)
            os.remove(chunk_file)
            
            # Unpack the segments and diarize them individually. 
            utterance_num= 1
            
            for segment in segments:
                segment_length= segment.end- segment.start # Length of the segment. Used for diarizer.
                accumulated_transcripts+= segment.text+" "
                
                # Find the closest matching speaker of this segment
                audio_to_diarize=  extract_audio_between_timestamps(recorded_audio, SAMPLE_RATE, segment.start, segment.end) # Extract the audio using timeframe
                try: # Sometimes the segement length would to too short to diarize and fails. So, this exception function is used to skip such cases.
                    diarizer_result = diarizer.getResults(audio_to_diarize, stepSize= segment_length)
                except:
                    continue
                speaker= list(diarizer_result.values())[0][0]

                # Prepare the output dictionary
                chunk_results = {"sessionID": sessionID,
                                "chunk_number":chunk_number,
                                "start_sec":segment.start,
                                "end_sec":segment.end
                                }
                unique_speakers= [speaker] # Will always have 1 speaker since the dictionary is at segment level and each segment will be associated to only one speaker.
                utterances= [{'utterance_id': utterance_num,
                              'text': segment.text,
                              'speaker': speaker,
                              'start_time': segment.start,
                              'end_time': segment.end,
                              'confidence': segment.avg_logprob
                              }]      
                chunk_results['unique_speakers']= unique_speakers
                chunk_results['utterances']= utterances   

                print(chunk_results)
                print(speaker,":", segment.text)
                print("----------------------------------------------")
                utterance_num+=1 

            chunk_number+=1     
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