from .vitl import VITL
from ..model import DecomposableIdentity
from .utils import ploss_with_crossing

class IQR(VITL):
    """Implements Infinite Quantile Regression as proposed in
    'Emotion Transfer Using Vector-Valued Infinite Task Learning'
    """

    def __init__(self, model, lbda, lbda_cross, sampler):
        super().__init__(model, ploss_with_crossing(lbda_cross), lbda, sampler)
