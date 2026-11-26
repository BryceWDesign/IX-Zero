"""Small deterministic CPU neural networks; no external runtime dependency."""
from math import exp, isfinite, sqrt, tanh
from random import Random


def sigmoid(x):
    return 1.0 / (1.0 + exp(-max(-40.0, min(40.0, x))))


class Dense:
    """One hidden tanh layer, sigmoid outputs, binary cross-entropy SGD.

    Inputs and targets are bounded; gradients use pre-update output weights.
    This implementation is intended for small experimental models, not scale.
    """
    def __init__(self, inputs, hidden, outputs, seed):
        if any(type(v) is not int or not 1 <= v <= 128 for v in (inputs, hidden, outputs)):
            raise ValueError('invalid network dimensions')
        self.shape = inputs, hidden, outputs
        rng = Random(seed)
        self.w1 = [[rng.uniform(-1, 1) / sqrt(inputs) for _ in range(inputs+1)] for _ in range(hidden)]
        self.w2 = [[rng.uniform(-1, 1) / sqrt(hidden) for _ in range(hidden+1)] for _ in range(outputs)]
        self.updates = 0

    @staticmethod
    def vector(values, n):
        if len(values) != n or any(type(x) not in (int, float) or not isfinite(x) or not 0 <= x <= 1 for x in values):
            raise ValueError('invalid normalized vector')
        return tuple(values)

    def forward(self, x):
        x = self.vector(x, self.shape[0])
        h = [tanh(sum(a*b for a,b in zip(w[:-1],x))+w[-1]) for w in self.w1]
        y = [sigmoid(sum(a*b for a,b in zip(w[:-1],h))+w[-1]) for w in self.w2]
        return h, y

    def predict(self, x):
        return self.forward(x)[1]

    def train(self, x, target, rate=0.08):
        x = self.vector(x, self.shape[0]); target = self.vector(target, self.shape[2])
        if not 0 < rate <= 1: raise ValueError('invalid learning rate')
        h, y = self.forward(x)
        delta = [p-t for p,t in zip(y,target)]
        dh = [(1-v*v)*sum(d*w[j] for d,w in zip(delta,self.w2)) for j,v in enumerate(h)]
        for w,d in zip(self.w2,delta):
            for j,v in enumerate(h+[1.0]): w[j] -= rate*d*v
        for w,d in zip(self.w1,dh):
            for j,v in enumerate((*x,1.0)): w[j] -= rate*d*v
        self.updates += 1

    def state(self):
        return dict(shape=list(self.shape), w1=[r[:] for r in self.w1], w2=[r[:] for r in self.w2], updates=self.updates)
