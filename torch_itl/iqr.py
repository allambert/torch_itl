from .estimator import ITLEstimator
from .cost import ploss_with_crossing
from .kernel import
from .model import KernelModel

def IQRegressor(ITLEstimator):

    def __init__(self, model = lbda_nc=0.01, lbda= 0.001,):
        super(IQRegressor, self).__init__(model, ploss_with_crossing(lbda_nc),
                                          lbda, sampler)
