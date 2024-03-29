"""
This module contains the SpeakerDiarizationChunkEcapa class, which is used to perform speaker diarization using the ECAPA-TDNN model.

The file is adapted from the SpeakerDiarizationChunk class in speaker_diarization_chunk.py.

Useful links:
https://speechbrain.readthedocs.io/en/latest/API/speechbrain.pretrained.interfaces.html#speechbrain.pretrained.interfaces.Pretrained
https://github.com/speechbrain/speechbrain/issues/574#issuecomment-803463181

"""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
#from speechbrain.pretrained import EncoderClassifier
from speechbrain.inference.speaker import EncoderClassifier
# from diarization.speaker_diarization_chunk import SpeakerDiarization
from sklearn.metrics.pairwise import cosine_similarity

from diarization.voice_activity_detection import voice_activity_detection


def pairwiseDists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Calculates pairwise distances between two matrices A and B using L2 norm, specifically for the ECAPA-TDNN model.

    Args:
        A (np.ndarray): Input matrix A of shape (L, 1, 192), where L is the number of segments in the input audio. O.1s segments are used.
        B (np.ndarray): Input matrix B of shape (N, 1, 192), where N is the number of reference speakers.

    Returns:
        np.ndarray: Pairwise distances matrix of shape (L, N)
    """

    B = B.reshape(1, B.shape[0], B.shape[2])
    D = A - B    # Compute the L2 norm along the last dimension to get pairwise distances
    return torch.norm(D, p=2, dim=2)

def pairwise_cosine_similarity(A, B):
    """Calculates pairwise distances between two matrices A and B using L2 norm, specifically for the ECAPA-TDNN model.

    Args:
        A (np.ndarray): Input matrix A of shape (L, 1, 192), where L is the number of segments in the input audio. O.1s segments are used.
        B (np.ndarray): Input matrix B of shape (N, 1, 192), where N is the number of reference speakers.

    Returns:
        np.ndarray: Pairwise distances matrix of shape (L, N)
    """

    # Convert the 3 demension Tensor to 2 dimension.
    A=A.squeeze(1)
    B=B.squeeze(1)
    
    # Calculate the cosine similarity
    cos_sim= cosine_similarity(A.cpu(),B.cpu()) # Add .cpu() to copy the tensor to host memory to convert to numpy
    cos_sim= torch.Tensor(cos_sim)
    return cos_sim


class SpeakerDiarizationChunkEcapa():
    def __init__(
        self,
        refSpeakers: dict,
        checkpoint_path: str = os.path.join("diarization", "checkpoints", "ecapa"),
        hparams_path: str = "inference.yaml",
        cache_path: str = os.path.join("diarization", "checkpoints", "ecapa", "cache"),
        sampleRate: int = 16000,
        vad: bool = False,
        stepSize: float = 0.1,
        device: torch.device = None,
    ):
        """
        Initialize the speaker diarization class with ECAPA model.

        Args:
            refSpeakers (dict): A dictionary of reference speaker IDs and audio data.
            checkpoint_path (str): Path to the model checkpoint.
            hparams_path (str): Path to the hyperparameters file.
            cache_path (str): Path to the cache directory.
            sampleRate (int): The sample rate of the audio data. (Optional, defaults to 16000)
            vad (bool): A flag to indicate if voice activity detection should be used. (Optional, defaults to False)
            stepSize (float): The step size for processing audio chunks. (Optional, defaults to 0.1)
            device (torch.device): The device to run the model on. Defaults to CPU.
        """
        self.sampleRate = sampleRate
        self.stepSize = stepSize
        self.vad = vad

        # Set the device for computation. Use the given device or default to the global setting.
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load the ECAPA model from the checkpoint and hyperparameters file
        self.encoder = EncoderClassifier.from_hparams(
            source=checkpoint_path,
            hparams_file=hparams_path,
            savedir=cache_path,
            run_opts={"device": self.device},
            overrides={"pretrained_path": checkpoint_path},
        )
        self.encoder.eval()

        # Create cluster centroids based on reference speaker samples
        self.clusters = []
        self.label = {}
        for count, (speaker_id, audio) in enumerate(refSpeakers.items()):
            self.label[count] = (speaker_id,)
            ref_data = self.getData(audio)
            ref_embs = self.getEmb(ref_data, mean=True)
            self.clusters.append(ref_embs)
        self.clusters = torch.stack(self.clusters)
        self.clustersNorm = F.normalize(self.clusters)

    def getEmb(
        self, data: np.ndarray, mean: bool = False, padding: bool = False, stepSize= None
    ) -> torch.Tensor:
        """
        Extracts speaker embeddings from the input audio data using the encoder.

        Args:
            data (np.ndarray): The input audio data to encode.
            mean (bool, optional): Whether to return the mean of the extracted embeddings. Defaults to False.
            padding (bool, optional): Whether to pad the audio data before processing. Defaults to False.

        Returns:
            torch.Tensor: The extracted speaker embeddings.
        """
        if (stepSize==None): stepSize= self.stepSize
        
        if padding:
            data = np.pad(
                data,
                (self.sampleRate, self.sampleRate),
                mode="constant",
                constant_values=0,
            )
        datatensor = []
        for i in range(int(data.shape[0] / self.sampleRate / stepSize - 2 / stepSize)):
            datatensor.append( data[ int(i * stepSize * self.sampleRate) : int(i * stepSize * self.sampleRate) + 2 * self.sampleRate])

        #print("------------ Datatensor Info --------------")
        #print(type(datatensor), len(datatensor), end=" ")
        #print(len(datatensor[0]))
        datatensor = np.stack(datatensor)
        datatensor = torch.from_numpy(datatensor).float().to(self.device)
        with torch.no_grad():
            embs = self.encoder.encode_batch(
                datatensor
            )  # Expect input shape (batch, features), expect output shape (batch, 1, 192)
        if mean:
            embs = torch.mean(embs, 0)
        return embs

    def getResults(self, audio: np.ndarray, stepSize= None) -> OrderedDict: # TODO: write a separate method for chunk results versus single audio results   
        """
        Processes the input audio data to obtain speaker diarization results.

        Args:
            audio (np.ndarray): The input audio data to diarize.

        Returns:
            OrderedDict: An OrderedDict containing timestamps and corresponding speaker labels, representing the diarization results.
        """
        if stepSize== None:
            stepSize= self.stepSize
        
        data = self.getData((audio, len(audio)))
        if len(audio) == 0:
            return OrderedDict()
        embs = self.getEmb(data, padding=True, stepSize= stepSize)
        embsNorm = F.normalize(embs)

        """
        print("------------------------------ embsNorm ----------------------------------------")
        print("Shape:", len(embsNorm), len(embsNorm))
        print("Type", type(embsNorm))
        print("Print",embsNorm)

        print("------------------------------ embsNorm ----------------------------------------")
        print("Shape:", len(self.clustersNorm), len(self.clustersNorm))
        print("Type", type(self.clustersNorm))
        print("Print",self.clustersNorm)
        """
        #dists = pairwiseDists(embsNorm, self.clustersNorm)
        dists= pairwise_cosine_similarity(embsNorm, self.clustersNorm)
        
        #print("--------------------Dist----------------------")
        #print(dists)
        #preds = torch.argmin(dists, dim=1)
        preds = torch.argmax(dists, dim=1)
        #print("--------------------Preds----------------------")
        #print(preds)
                

        # get vad results if vad flag is true
        if self.vad:
            vadRes = np.array(
                voice_activity_detection(data, self.sampleRate, 1)
            ).astype(int)
            numSegments = int(self.stepSize * 1000 / 20)
            padLength = numSegments - vadRes.shape[0] % numSegments
            vadRes = np.pad(vadRes, (0, padLength), mode="symmetric")
            vadRes = np.reshape(vadRes, (-1, numSegments))
            vadRes = np.sum(vadRes, 1) > 2

        result = OrderedDict()
        for pCnt, p in enumerate(preds):
            timestamp = pCnt * self.stepSize
            label = self.label[p.item()]
            if self.vad:
                if vadRes[pCnt]:
                    result["{:0.1f}".format(timestamp)] = label
            else:
                result["{:0.1f}".format(timestamp)] = label

        return result
