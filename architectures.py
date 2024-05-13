from flax import linen as nn

class MLP(nn.Module):
    """Three-layer MLP as used for Example 2."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=500)(x)
        x = nn.softplus(x)
        x = nn.Dense(features=500)(x)
        x = nn.softplus(x)
        x = nn.Dense(features=2)(x)
        return x
  

class CNN_celebA(nn.Module):
    """Five-layer CNN as outlined in Table C.1."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.001)
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.001)
        x = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.001)
        x = nn.Conv(features=128, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.001)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=2)(x)
        return x


class CNN_mnist(nn.Module):
    """Three-layer CNN as outlined in Table C.1."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x
