from faster_whisper import WhisperModel
import pyaudio
import wave
import os

def record_chunk(p, stream, file_path, chunk_length=3):
    frames=[]
    for _ in range(0, int(16000/1024*chunk_length)):
        data= stream.read(1024)
        frames.append(data)
    
    wf= wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_chunk(model, chunk_file):
    segments, info= model.transcribe(chunk_file, beam_size= 5)
    transcription=""
    for segment in segments:
        transcription+= segment.text
    return transcription

def main():
    model_size= "large-v3"
    #model_size= "medium"
    #model_size = "distil-large-v2"
    #model_size= "distil-medium.en"
    model= WhisperModel(model_size, device= "cuda", compute_type= "float16")
    #model= WhisperModel(model_size, device= "cpu", compute_type= "int8")

    p= pyaudio.PyAudio()
    stream= p.open(format= pyaudio.paInt16, channels=1, rate= 16000, input=True, frames_per_buffer= 1024)
    print("Recording Started...")

    accumulated_transcripts= ""

    try:
        while True:
            chunk_file= 'temp_chunk2.wav'
            record_chunk(p, stream, chunk_file)
            #print("recording works")
            transcription= transcribe_chunk(model, chunk_file)
            #print("transcription works")
            print(transcription)
            os.remove(chunk_file)

            accumulated_transcripts+= transcription+" "
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