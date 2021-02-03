import torch
import numpy as np

emo_va = {'Neutral': [0, 0],
          'Happy': [0.6647930758154823, 0.07025315235958239],
          'Sad': [-0.6364936709598201, -0.25688320447600566],
          'Surprise': [0.17960005233680493, 0.6894792038743631],
          'Fear': [-0.1253752865553082, 0.7655788112100937],
          'Disgust': [-0.6943645673378837, 0.457145871269001],
          'Anger': [-0.452336028803629, 0.5656012294430937],
          'Contempt': [-0.5138537435929467, 0.5825553992724378]}


class CircularEmoSampler(object):

    def __init__(self, inp_emotion='Neutral', inc_emotion=True):
        self.inp_emotion = inp_emotion
        self.inc_emotion = inc_emotion
        self.emo_list = ['Anger', 'Disgust', 'Fear',
                         'Happy', 'Sad', 'Surprise', 'Neutral']

    def sample(self):
        emo_emb = []
        for emo in self.emo_list:
            if (not self.inc_emotion) and (emo == self.inp_emotion):
                continue
            else:
                if emo == 'Neutral':
                    emo_emb.append(emo_va['Neutral'])
                else:
                    emo_emb.append(emo_va[emo] / np.linalg.norm(emo_va[emo]))
        return torch.tensor(emo_emb, dtype=torch.float)
