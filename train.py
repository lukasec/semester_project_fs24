from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


class AutoEncoder(nn.Module):
    encoder_widths: Sequence[int]
    decoder_widths: Sequence[int]
    input_shape: Sequence[int]

    def setup(self):
        input_dim = np.prod(self.input_shape)
        self.encoder = MLP(self.encoder_widths)
        self.decoder = MLP(self.decoder_widths + (input_dim,))

    def __call__(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        assert x.shape[1:] == self.input_shape
        return self.encoder(jnp.reshape(x, (x.shape[0], -1)))

    def decode(self, z):
        z = self.decoder(z)
        x = nn.sigmoid(z)
        x = jnp.reshape(x, (x.shape[0],) + self.input_shape)
        return x


model = AutoEncoder(encoder_widths=[20, 10, 5],
                    decoder_widths=[5, 10, 20],
                    input_shape=(12,))
batch = jnp.ones((16, 12))
variables = model.init(jax.random.key(0), batch)
encoded = model.apply(variables, batch, method=model.encode)
decoded = model.apply(variables, encoded, method=model.decode)
print(decoded)
