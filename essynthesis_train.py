import torch
from datasets.datasets import import_speech_synth_ravdess
from torch_itl import model, sampler, cost, kernel, estimator


print('Reading data')
x_train, y_train = import_speech_synth_ravdess()
n = x_train.shape[0]
m = y_train.shape[1]
nf = y_train.shape[2]
print('data dimensions', n, m, nf)

print('defining network')
# define input network and kernel
n_h = 40
d_out = 10
model_kernel_input = torch.nn.Sequential(
    torch.nn.Linear(x_train.shape[1], n_h),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(n_h, n_h),
    torch.nn.Linear(n_h, d_out),
)

gamma = 3
optim_params = dict(lr=0.0001, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False)

kernel_input = kernel.LearnableGaussian(
    gamma, model_kernel_input, optim_params)


# define emotion kernel
kernel_output = kernel.Gaussian(1.0)

# define model
itl_model = model.SpeechSynthesisKernelModel(kernel_input, kernel_output,
                                             kernel_freq=torch.eye(nf, dtype=torch.float))

# define cost function
cost_function = cost.speech_synth_loss
lbda = 0.001

# define emotion sampler
sampler_ = sampler.CircularSampler()
sampler_.m = m


itl_estimator = estimator.ITLEstimator(itl_model,
                                       cost_function, lbda, sampler_)

itl_estimator.fit_alpha(x_train, y_train, n_epochs=10,
                        lr=0.001, line_search_fn='strong_wolfe')

print(itl_estimator.losses)
itl_estimator.clear_memory()
itl_estimator.fit_kernel_input(x_train, y_train, n_epochs=10)
print(itl_estimator.model.kernel_input.losses)
itl_estimator.model.kernel_input.clear_memory()
