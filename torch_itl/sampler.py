import torch
import numpy as np


class LinearSampler(object):
    'Linear sampling class'

    def __init__(self, a, b, m=None, epsilon=None):
        self.a = a
        self.b = b
        if epsilon is None:
            self.epsilon = 0.01
        else:
            self.epsilon = epsilon

    def sample(self, m=None):
        if m is None:
            if not hasattr(self, 'm'):
                raise ValueError('No value given for m')
            else:
                return torch.linspace(self.a + self.epsilon, self.b - self.epsilon, self.m).view(-1, 1)
        else:
            return torch.linspace(self.a + self.epsilon, self.b - self.epsilon, m).view(-1, 1)


# TODO: Currently hand coded values are returned, modify class later
class CircularSampler(object):
    def __init__(self, a=0, b=torch.acos(torch.zeros(1)).item()*4, m=None, data=None, sample_dict=None,
                 inp_emotion='Neutral', inc_emotion=False):
        self.a = a
        self.b = b
        self.m = m
        self.data = data
        self.sample_dict = sample_dict
        self.sample_dict['Neutral'] = [0, 0]
        self.inp_emotion = inp_emotion
        self.inc_emotion = inc_emotion

        if data == 'KDEFaff':
            self.emo_list = ['Fear', 'Anger', 'Disgust', 'Happy', 'Sad', 'Surprise', 'Neutral']
        elif data == 'Rafdaff':
            self.emo_list = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def sample(self, m):
        # angles = torch.linspace(a, b, m)
        # return torch.stack((torch.cos(angles), torch.sin(angles))).T

        # Just for now, to avoid errors and keep the order in check
        assert (m == self.m)
        if self.data == 'KDEF':
            return torch.tensor([[-0.259, 0.966], [-0.707, 0.707], [-0.866, 0.5],
                                 [1, 0], [-1, 0], [0.259, 0.966]], dtype=torch.float)
        elif self.data == 'ravdess':
            return torch.tensor([[1/2, -(3**0.5)/2], [1, 0], [-1, 0], [1/2**0.5, 1/2**0.5],
                                 [0, 1]], dtype=torch.float)
        elif self.data == 'KDEFaff' or self.data == 'Rafdaff':
            emo_emb = []
            for emo in self.emo_list:
                if (not self.inc_emotion) and (emo == self.inp_emotion):
                    continue
                else:
                    if emo == 'Neutral':
                        emo_emb.append(self.sample_dict['Neutral'])
                    else:
                        emo_emb.append(self.sample_dict[emo]/np.linalg.norm(self.sample_dict[emo]))
            return torch.tensor(emo_emb, dtype=torch.float)

        else:
            return torch.from_numpy(np.expand_dims(np.linspace(start=-np.pi/2, stop=np.pi/2, num=m), axis=-1)).float()
