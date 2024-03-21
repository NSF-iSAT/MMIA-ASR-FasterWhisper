import webrtcvad
import numpy as np

def int_to_pcm16(audio):
    ints = audio.astype(np.int16)
    little_endian = ints.astype('<u2')
    buf = little_endian.tostring()
    return buf

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def voice_activity_detection(data, sample_rate, aggressiveness):
    audio = int_to_pcm16(data)
    vad = webrtcvad.Vad(aggressiveness)
    frames = frame_generator(20, audio, sample_rate)
    frames = list(frames)
    res = []
    for frame in frames:
        res.append(vad.is_speech(frame.bytes, sample_rate))
    return res
