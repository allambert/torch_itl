import torch
from datasets.datasets import import_speech_synth_ravdess
from torch_itl import model, sampler, cost, kernel, estimator


print('Reading data')
x_train, y_train, x_mean = import_speech_synth_ravdess()
print(x_train.shape)
n = x_train.shape[0]
m = y_train.shape[1]
nf = y_train.shape[2]
print('data dimensions', n, m, nf)

print('defining network')
# define input network and kernel
n_h = 32
d_out = 10


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


# model_kernel_input = torch.nn.Sequential(
#     torch.nn.Conv2d(1, 16, 2),
#     torch.nn.ReLU(),
#     torch.nn.Conv2d(16, 32, 2),
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d((78, 9), stride=1),
#     Flatten(),
#     torch.nn.Linear(n_h, d_out),
# )

model_kernel_input = torch.nn.Sequential(
   torch.nn.Linear(x_train.shape[1], n_h),
   torch.nn.LeakyReLU(),
   torch.nn.Linear(n_h, d_out),
)

gamma = 3
optim_params = dict(lr=0.01, momentum=0, dampening=0,
                    weight_decay=0.0, nesterov=False)

kernel_input = kernel.LearnableGaussian(
    gamma, model_kernel_input, optim_params)


# kernel_input = kernel.Gaussian(3.0)

# define emotion kernel
kernel_output = kernel.Gaussian(1.0)

# define model
itl_model = model.SpeechSynthesisKernelModel(kernel_input, kernel_output,
                                             kernel_freq=torch.eye(nf, dtype=torch.float))

# define cost function
cost_function = cost.speech_synth_loss
lbda = 0.01

# define emotion sampler
sampler_ = sampler.CircularSampler(data='kdef')
sampler_.m = m


itl_estimator = estimator.ITLEstimator(itl_model,
                                       cost_function, lbda, sampler_)

NE = 10
for ne in range(NE):
    # itl_estimator.fit_alpha(x_train, y_train, n_epochs=2, solver='lbfgs',
    #                     lr=0.1)
    itl_estimator.fit_alpha(x_train, y_train, n_epochs=5, solver='adam',
                           lr=0.1)
    print(itl_estimator.losses)
    itl_estimator.clear_memory()
    itl_estimator.fit_kernel_input(x_train, y_train, n_epochs=5)
    print(itl_estimator.model.kernel_input.losses)
    itl_estimator.model.kernel_input.clear_memory()

print('almost done!')



"""
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

model_kernel_input = torch.nn.Sequential(
    torch.nn.Conv2d(1, 16, 2),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 32, 2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d((78, 9), stride=1),
    Flatten(),
    torch.nn.Linear(n_h, d_out),
)

import numpy as np
from IPython.display import Audio

pred = np.load('pred2.npy').T
print(pred.shape)

mel = torch.from_numpy(np.expand_dims(pred, 0)).float().to('cuda')
with torch.no_grad():
  audio = waveglow.infer(mel)
audio_numpy = audio[0].data.cpu().numpy()

rate = 22050
Audio(audio_numpy, rate=rate)

"""
