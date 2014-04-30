import theano.tensor as T
from theano.compat.python2x import OrderedDict
from pylearn2.utils import wraps
from pylearn2.training_algorithms.learning_rule import LearningRule


class CensorLearningRule(LearningRule):
    """
    A learning rule which appliess a series of censorships to the gradients
    and applies a user-defined learning rule.

    Parameters
    ----------
    censors : list
        Censorship objects
    learning_rule : LearningRule
        User-defined learning rule to apply
    """
    def __init__(self, censors, learning_rule):
        self.censors = censors
        self.learning_rule = learning_rule

    @wraps(LearningRule.add_channels_to_monitor)
    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        self.learning_rule.add_channels_to_monitor(monitor, monitoring_dataset)

    @wraps(LearningRule.get_updates)
    def get_updates(self, learning_rate, grads, lr_scalers=None):
        for censor in self.censors:
            grads = censor.censor_gradients(grads)
        return self.learning_rule.get_updates(learning_rate, grads, lr_scalers)


class GradientClipping:
    """
    Clips gradients when they exceed a certain value.
    """
    def __init__(self, max_magnitude, rescale=False):
        self.max_magnitude = max_magnitude
        self.rescale = rescale

    def censor_gradients(self, grads):
        censored_grads = OrderedDict()
        if self.rescale:
            norm = 0
            for grad in grads.values():
                norm += (grad ** 2).sum()
            norm = T.switch(T.sqrt(norm) > self.max_magnitude,
                            self.max_magnitude / T.sqrt(norm), 1.)
            for param, grad in grads.items():
                censored_grads[param] = grad * norm
        else:
            for param, grad in grads.items():
                censored_grads[param] = T.clip(grad,
                                               -self.max_magnitude,
                                               self.max_magnitude)
        return censored_grads
