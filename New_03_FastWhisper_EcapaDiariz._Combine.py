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



def process_asr_and_diarizer(diarizer_result, asr_result, chunk_metadata:dict, CHUNK_LENGTH): #chunk_path:str,
    """process a chunk

    Args:
        chunk_path (str): path to chunk (on local filesystem)
        diarizer_result (dict): An ordered dictionary with start time and speaker ID 
        asr_result (str): Contain the transcript of this segment 
        chunk_metadata (dict): chunk info with fields ['sessionID','chunk_number','lesson','groupid',...] #TODO: decide on metadata 

    returns:
        chunk_results: _description_ #TODO: do we still need it to bea generator or can it just be a regular return value
    """    
    #chunk, chunk_len_samp = load_video_to_ndarray(chunk_path) 

    # ASR setup

    prev_chunk = None  # ASR will use previous chunk to catch words on chunk boundary
    prev_asr_result = None  # ASR will use previous ASRresult to catch words on chunk boundary

    # Output dict setup
    #min_chunk_samp = SAMPLE_RATE * MIN_CHUNK_LEN
    utterance_num = 0


    #print(f"\nProcessing video from {chunk_metadata["sessionID"]}, chunk {chunk_metadata['chunk_number']} ...\n")

    # Output dict setup
    speaker_list = []

    chunk_start_sec = float(chunk_metadata.get('start_sec', 0))#TODO: Aravind: make sure chunk_metadata contains key 'start_sec'
    chunk_end_sec = float(chunk_metadata.get('end_sec', CHUNK_LENGTH))# Aravind: make sure chunk_metadata contains key 'end_sec'

    chunk_results = chunk_metadata # copy some fields from the input chunk metadata

    if len(diarizer_result) == 0:
        chunk_results['unique_speakers'] = []
        chunk_results['signal_to_noise'] = 0
        chunk_results['utterances'] = {}
        return chunk_results

    reformatted_diar_result = {(chunk_start_sec + float(key)): ','.join(value) for key, value in diarizer_result.items()} # change from chunk timing to overall timing
    #print("------------------------- reformatted_diar_result -------------------------------------------")
    #print(reformatted_diar_result)

    # ASR
    reformatted_asr_result = {(chunk_start_sec + round(float(key), 1)): value for key, value in asr_result.items()} # change from chunk timing to overall timing

    #reformatted_asr_result= {chunk_start_sec: asr_result}
    #print("------------------------- reformatted_asr_result -------------------------------------------")
    #print(reformatted_asr_result)

    # print('asr result is', asr_result)
    # Merge Diarizer & ASR Output

    # TODO: Rosy: Update implementation to assign words to a speaker when the diarizer has no output,
    #  but the same speaker is recorded before and after the gap (see ASR.BTS_ASR.mergeDiarizedASR)

    merged_diar_asr_result = []
    partial_utterance = []
    confidences = []
    utterance_start_time = None
    utterance_end_time = None
    prev_speaker = None
    for timestamp, data in reformatted_asr_result.items():
        if timestamp in reformatted_diar_result:  # O(1) lookup
            speaker = reformatted_diar_result[timestamp]
        else:
            speaker = 'undefined'

        if speaker not in speaker_list:
            speaker_list.append(speaker)

        if prev_speaker and prev_speaker == speaker:
            # continue concatenating output for this utterance as this word is from the same speaker as previous word
            partial_utterance.append(data['word'])
            confidences.append(data['confidence'])
            utterance_end_time = round(chunk_start_sec + data['end_time'], 1)
        else:
            if prev_speaker:
                # end previous utterance
                merged_diar_asr_result.append({'utterance_id': utterance_num,
                                                'text': " ".join(partial_utterance),
                                                'speaker': prev_speaker,
                                                'start_time': utterance_start_time,
                                                'end_time': utterance_end_time,
                                                'confidence': round(np.mean(confidences), 5)})
                utterance_num += 1

            # start new utterance
            partial_utterance = [data['word']]
            confidences = [data['confidence']]
            utterance_start_time = timestamp
            utterance_end_time = round(chunk_start_sec + data['end_time'], 1)

        prev_speaker = speaker

    if partial_utterance:
        # start the anonymization process
        anonymized_text = anonymize_text(" ".join(partial_utterance))
        merged_diar_asr_result.append({'utterance_id': utterance_num,
                                        'text': anonymized_text,
                                        'speaker': prev_speaker,
                                        'start_time': utterance_start_time,
                                        'end_time': utterance_end_time,
                                        'confidence': round(np.mean(confidences), 5)})
        utterance_num += 1
    # Final chunk results to be passed to inference
    session_id = chunk_metadata["sessionID"]

    # SNR
    #snr_result = wada_snr(chunk)

    # Final chunk results to be passed to inference
    chunk_results["sessionID"] = session_id
    #chunk_results['signal_to_noise'] = snr_result
    chunk_results['unique_speakers'] = speaker_list
    chunk_results['utterances'] = json.loads(json.dumps(merged_diar_asr_result), parse_float=Decimal)
    return chunk_results


def main():

    # Initialize the fast whisper models
    model_size= "large-v3" #"medium" "distil-large-v2"
    model= WhisperModel(model_size, device= str(device), compute_type= "float16")

    # Start recording the audio
    SAMPLE_RATE = 16000  # Hz, see load_video for requirements
    CHUNK_LENGTH = 3  # Define the frequency of recording in seconds
    DIARIZE_STEP_SIZE= 1 # Define the frequency of diarization in seconds
    MIN_CHUNK_LEN = 0.1 # seconds, chunks shorter than this will be skipped
    p= pyaudio.PyAudio()
    stream= p.open(format= pyaudio.paInt16, channels=1, rate= SAMPLE_RATE, input=True, frames_per_buffer= 1024)
    print("Recording Started...")

    accumulated_transcripts= ""

    # Initialze the diarizer with recorded enrollments
    enrollment_utterances = {'Niranjan':load_video_to_ndarray('Enrollment_Niranjan.wav'),
                             'Kamala Harris':load_video_to_ndarray('Enrollment_Kamala_Harris.wav'), 
                             'Trump':load_video_to_ndarray('Enrollment_Trump.wav')}
    diarizer = SpeakerDiarization(refSpeakers=enrollment_utterances, stepSize=DIARIZE_STEP_SIZE) #stepSize decides the frequency of diarization. Say if it is 0.5 then the result will be given as a speaker ID for every 0.5secs of the input audio in a dictionary.

    try:
        while True:
            # Retrive the recorded audio and transcribe them
            chunk_file= 'temp_chunk.wav'
            recorded_audio= record_chunk(p, stream, chunk_file, chunk_length=CHUNK_LENGTH)
            segments, info= model.transcribe(chunk_file, beam_size= 5)

            os.remove(chunk_file)

            sessionID= 0
            chunk_number= 0
            # Unpack the segments and diarize them individually. 

            
            for segment in segments:
                #print("Segment entering")
                seg_start_time = segment.start # Start time of the segment
                seg_end_time = segment.end # End time of the segment
                text = segment.text  # Transcribed text of the segment
                confidence= segment.avg_logprob
                accumulated_transcripts+= text+" "
                print(seg_start_time, "to", seg_end_time)
                
                # Calculate the closest matching speaker in the segment
                audio_to_diarize=  extract_audio_between_timestamps(recorded_audio, SAMPLE_RATE, seg_start_time, seg_end_time) # Extract the audio using timeframe
                #print(len(audio_to_diarize))

                chunk_metadata = {"sessionID": sessionID,
                                "chunk_number":chunk_number,
                                "start_sec":seg_start_time,
                                "end_sec":seg_end_time
                                }
                sessionID+=1
                chunk_number+=1

                diarizer_result = diarizer.getResults(audio_to_diarize)
                #print("--------------------- diarizer_result -------------------------------")
                print(diarizer_result)

                asr_result= {seg_start_time:{'word': text, 'confidence':confidence, 'end_time':seg_end_time}}
                chunk_results= process_asr_and_diarizer(diarizer_result, asr_result= asr_result, chunk_metadata= chunk_metadata, CHUNK_LENGTH= CHUNK_LENGTH)
                #chunk_results= process_asr_and_diarizer(diarizer_result, asr_result= segments, chunk_metadata= chunk_metadata, CHUNK_LENGTH= CHUNK_LENGTH)
                #print("--------------------- chunk_results -------------------------------")
                print(chunk_results)
                """
                try:
                    diarizer_result = diarizer.getResults(audio_to_diarize)
                    print("--------------------- diarizer_result -------------------------------")
                    print(diarizer_result)
                    
                    chunk_results= process_asr_and_diarizer(diarizer_result, asr_result= text, chunk_metadata= chunk_metadata)
                    print("--------------------- chunk_results -------------------------------")
                    print(chunk_results)

                """
                """
                    speaker= [i for i in diarizer_result.values()][0][0]
                    print("--------------------- speaker 1 -------------------------------")
                    print(speaker)
                    speaker= diarizer_result['0.0'][0] # This is just taking the first detected speaker, rather just the mode of speakers.
                    print("--------------------- speaker 2 -------------------------------")
                    #print("Speaker:",speaker)
                    print(speaker,":", text) 
                """
                """
                except:
                    continue
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