import torch
from pyannote.audio.models import SincTDNN
from pyannote.audio.train.task import Task, TaskOutput, TaskType
from scipy.io import wavfile
import numpy as np
import torch.nn as nn
import os
from itertools import combinations
from collections import OrderedDict
import torch.nn.functional as F
from diarization.voice_activity_detection import voice_activity_detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return torch.norm(D, p=2, dim=2)

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
        data = self.getData((audio, len(audio)))
        if(len(audio) == 0):
            return OrderedDict()
        embs = self.getEmb(data, padding=True)
        embsNorm = F.normalize(embs)
        dists = pairwiseDists(embsNorm, self.clustersNorm)
        preds = torch.argmin(dists, 1)

        # get vad results if vad flag is true
        if self.vad:
            vadRes = np.array(voice_activity_detection(data, self.sampleRate, 1)).astype(int)
            numSegments = int(self.stepSize*1000/20)
            padLength = numSegments - vadRes.shape[0]%numSegments
            vadRes = np.pad(vadRes, (0, padLength), mode='symmetric')
            vadRes = np.reshape(vadRes, (-1, numSegments))
            vadRes = np.sum(vadRes, 1) > 2

        result = OrderedDict()
        for pCnt, p in enumerate(preds):
            timestamp = pCnt * self.stepSize
            label = self.label[p.item()]
            if self.vad:
                if vadRes[pCnt]:
                    result['{:0.1f}'.format(timestamp)] = label
            else:
                result['{:0.1f}'.format(timestamp)] = label
            
        return result


