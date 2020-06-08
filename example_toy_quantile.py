import torch
from operalib.datasets.quantile import toy_data_quantile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch_itl import model, sampler, cost, kernel

# %%

dtype = torch.float
device = torch.device("cpu")

# %%
# Defining a simple toy dataset:

print("Creating the dataset")

x_train, y_train, _ = toy_data_quantile(150)
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
n = x_train.shape[0]
m = 10

plt.figure()
plt.scatter(x_train, y_train, marker='.')
plt.show()

# %%
# Defining an ITL model, first without a learnable kernel

print("Defining the model")

kernel_input = kernel.Gaussian(3.5)
kernel_output = kernel.Gaussian(9)
cost_function = cost.ploss_with_crossing(0.01)
lbda = 0.001
sampler_ = sampler.LinearSampler(0.1, 0.9, 10, 0)
sampler_.m = 10

itl_model = model.ITL(kernel_input, kernel_output,
                      cost_function, lbda, sampler_)

# Learning the coefficients of the model
print("Fitting the coefficients of the model")

itl_model.fit_alpha(x_train, y_train, n_epochs=40,
                    lr=0.001, line_search_fn='strong_wolfe')

# Plotting the loss along learning

plt.figure()
plt.title("Loss evolution with time")
plt.plot(itl_model.losses)
plt.show()
best_loss = itl_model.losses[-1]

# Plotting the model on test points

probs = itl_model.sampler.sample(30)
x_test = torch.linspace(0, 1.4, 100).view(-1, 1)
y_pred = itl_model.forward(x_test, probs).detach().numpy()
colors = [cm.viridis(x.item()) for x in torch.linspace(0, 1, 30)]
plt.figure()
plt.title("Conditional Quantiles output by our model")
plt.scatter(x_train,y_train,marker='.')
for i in range(30):
    plt.plot(x_test, y_pred[:, i], c=colors[i])
plt.show()

# %%
# Let's learn the input kernel with ITL
# First define a neural net

n_h = 40
d_out = 10
model_kernel_input = torch.nn.Sequential(
    torch.nn.Linear(x_train.shape[1], n_h),
    torch.nn.ReLU(),
    torch.nn.Linear(n_h, n_h),
    torch.nn.Linear(n_h, d_out),
)
gamma = 3
optim_params = dict(lr=0.1, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False)

kernel_input = kernel.LearnableGaussian(gamma, model_kernel_input, optim_params)
itl_model.kernel_input = kernel_input

# %%

itl_model.fit_kernel_input(x_train,y_train)

# plot the loss along learning the kernel

plt.figure()
plt.title("Loss evolution when learning the kernel")
plt.plot(itl_model.kernel_input.losses)
plt.show()

# %%
# Now retrain the parameters alpha of the model

itl_model.clear_memory()
itl_model.fit_alpha(x_train,y_train,n_epochs=40,lr=0.01,line_search_fn='strong_wolfe')

# plot the loss

plt.figure()
plt.title("Loss evolution when learning model coefficients again")
plt.plot(itl_model.losses)
plt.show()

y_pred = itl_model.forward(x_test,probs).detach().numpy()
colors = [cm.viridis(x.item()) for x in torch.linspace(0, 1, 30)]
plt.figure()
plt.title('Conditional Quantiles with learned kernel')
plt.scatter(x_train,y_train,marker='.')
for i in range(30):
    plt.plot(x_test,y_pred[:,i],c=colors[i])
plt.show()

print('Loss gain from learning the kernel: ',best_loss - itl_model.losses[-1])
