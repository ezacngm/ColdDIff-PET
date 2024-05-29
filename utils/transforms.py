"""
Authors: Xingyu Xie
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch


class CircleMask(object):
    def __init__(self, sz=256, r=120):
        self.sz = sz
        self.r = r
        self.mask = self.make_mask()

    def __call__(self, tensor):
        return tensor * self.mask

    def __repr__(self):
        return self.__class__.__name__ + '(sz={256}, r={120})'.format(self.sz, self.r)

    def make_mask(self):
        # Create a grid of coordinates
        x, y = torch.meshgrid(torch.arange(self.sz), torch.arange(self.sz), indexing='ij')
        # Calculate the distance from the center and compare it with the radius
        return ((x - self.sz // 2) ** 2 + (y - self.sz // 2) ** 2 < self.r ** 2).float()


class UniformNoise(object):
    def __init__(self, a=-0.1, b=0.1, rate=0.2):
        self.a = a  # minimum in Uniform Distribution
        self.b = b  # maximum in Uniform Distribution
        self.rate = rate  # The proportion of uniform random events in the noise-free data

    def __call__(self, tensor):
        w_mul_noise = tensor * (torch.rand_like(tensor) * (self.b - self.a) + self.a + 1)
        results = w_mul_noise + self.rate * torch.mean(w_mul_noise, dim=(-1, -2), keepdims=True)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(a={self.a}, b={self.b}, rete={self.rate})'


class LevelCount(object):
    def __init__(self, count=250000):
        self.count = count

    def __call__(self, tensor):
        return tensor * self.count / (1e-9 + torch.sum(tensor, dim=(-1, -2), keepdim=True))

    def __repr__(self):
        return f'{self.__class__.__name__}(count={self.count})'


# Define a custom transform class for sampling from the Poisson distribution
class SamplePoisson(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return torch.poisson(tensor)

    def __repr__(self):
        return self.__class__.__name__

# Define a custom transform class for adding uniform noise
class AddUniformNoise(object):
    def __init__(self, low=0., high=1.):
        self.low = low
        self.high = high

    def __call__(self, tensor):
        return tensor + torch.rand(tensor.size()) * (self.high - self.low) + self.low

    def __repr__(self):
        return self.__class__.__name__ + '(low={0}, high={1})'.format(self.low, self.high)


class AddMultiplicativeNoise(object):
    def __init__(self, low=-0.1, high=0.1):
        self.low = low
        self.high = high

    def __call__(self, tensor):
        return tensor * (torch.rand_like(tensor) * (self.high - self.low) + self.low)

    def __repr__(self):
        return self.__class__.__name__ + '(low={-0.1}, high={0.1})'.format(self.low, self.high)


class AddAddictiveNoise(object):
    def __init__(self, low=-0.1, high=0.1, scale=0.2):
        self.low = low
        self.high = high
        self.scale = scale

    def __call__(self, tensor):
        return self.scale * torch.mean(tensor * (torch.rand_like(tensor) * (self.high - self.low) + self.low), dim=(-1, -2), keepdims=True)

    def __repr__(self):
        return self.__class__.__name__ + '(low={-0.1}, high={0.1}, scale={0.2})'.format(self.low, self.high, self.scale)

