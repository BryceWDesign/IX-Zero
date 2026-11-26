"""Bounded progressive recognition: frozen lateral features, trainable new head.

Parent networks are immutable copies. The trainable head can learn a new
sensor mapping without overwriting old weights. Depth and fan-in are bounded.
"""
from copy import deepcopy
from .neural import Dense

MAX_DEPTH = 3
MAX_PARENTS = 2


class ProgressiveRecognition:
    def __init__(self, parents, seed):
        if not 1 <= len(parents) <= MAX_PARENTS or any(not isinstance(p,(Dense,ProgressiveRecognition)) for p in parents):
            raise ValueError('invalid lateral sources')
        self.depth = 1+max(getattr(p,'depth',1) for p in parents)
        if self.depth > MAX_DEPTH:raise ValueError('progressive depth limit')
        self.parents=tuple(deepcopy(p) for p in parents)
        self.head=Dense(3+sum(p.shape[1] for p in parents),8,1,seed)
        self.shape=(3,8,1)

    def _features(self,x):
        x=Dense.vector(x,3)
        return (*x,*((v+1)/2 for p in self.parents for v in p.forward(x)[0]))

    def forward(self,x):return self.head.forward(self._features(x))
    def predict(self,x):return self.forward(x)[1]
    def train(self,x,y,rate=.08):self.head.train(self._features(x),y,rate)
    @property
    def updates(self):return self.head.updates
    def state(self):
        return dict(kind='progressive',depth=self.depth,shape=list(self.shape),updates=self.updates,
                    head=self.head.state(),parents=[p.state() for p in self.parents])
