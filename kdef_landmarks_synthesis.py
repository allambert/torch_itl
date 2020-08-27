import torch
import numpy as np
from datasets.datasets import import_kdef_landmark_synthesis
import matplotlib.pyplot as plt
from torch_itl import model, sampler, cost, kernel, estimator

# set some params
kernel_input_learnable = False
if kernel_input_learnable:
    NE = 10  # num epochs overall
    ne_fa = 5  # num epochs fit alpha per NE
    ne_fki = 5  # num epochs fit kernel per NE
else:
    NE = 1
    ne_fa = 50

print('Reading data')
x_train, y_train, x_test, y_test = import_kdef_landmark_synthesis(dtype='aligned')
n = x_train.shape[0]
m = y_train.shape[1]
nf = y_train.shape[2]
print('data dimensions', n, m, nf)

if kernel_input_learnable:
    print('defining network')
    # define input network and kernel
    n_h = 64
    d_out = 64
    model_kernel_input = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], n_h),
        torch.nn.ReLU(),
        torch.nn.Linear(n_h, d_out)
    )

    gamma = 3
    optim_params = dict(lr=0.01, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)

    kernel_input = kernel.LearnableGaussian(
        gamma, model_kernel_input, optim_params)
else:
    kernel_input = kernel.Gaussian(3.0)

# define emotion kernel
kernel_output = kernel.Gaussian(1.0)

# define model
itl_model = model.SpeechSynthesisKernelModel(kernel_input, kernel_output,
                                             kernel_freq=torch.eye(nf, dtype=torch.float))

# define cost function
cost_function = cost.speech_synth_loss
lbda = 0.001

# define emotion sampler
sampler_ = sampler.CircularSampler(data='kdef')
sampler_.m = m


itl_estimator = estimator.ITLEstimator(itl_model,
                                       cost_function, lbda, sampler_)

for ne in range(NE):
    itl_estimator.fit_alpha(x_train, y_train, n_epochs=ne_fa,
                        lr=0.01, line_search_fn='strong_wolfe')

    print(itl_estimator.losses)
    itl_estimator.clear_memory()
    if kernel_input_learnable:
        itl_estimator.fit_kernel_input(x_train, y_train, n_epochs=ne_fki)
        print(itl_estimator.model.kernel_input.losses)
        itl_estimator.model.kernel_input.clear_memory()


pred_test1 = itl_estimator.model.forward(x_test, sampler_.sample(m))
pred_test2 = itl_estimator.model.forward(x_test, torch.tensor([[0.866, 0.5]], dtype=torch.float))

