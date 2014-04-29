import theano.tensor as T
from theano.compat.python2x import OrderedDict
from pylearn2.utils import wraps
from pylearn2.training_algorithms.learning_rule import LearningRule


class ChainedLearningRule(LearningRule):
    """
    A learning rule which appliess a series of learning rules sequentially.

    Parameters
    ----------
    learning_rules : list of LearningRule
        List of learning rules to apply, with the first element of the list
        being applied first and the last element of the list being applied
        last.
    """
    def __init__(self, learning_rules):
        self.learning_rules = learning_rules

    @wraps(LearningRule.add_channels_to_monitor)
    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        for learning_rule in self.learning_rules:
            learning_rule.add_channels_to_monitor(monitor, monitoring_dataset)

    @wraps(LearningRule.get_updates)
    def get_updates(self, learning_rate, grads, lr_scalers=None):
        for learning_rule in self.learning_rules:
            grads = learning_rule.get_updates(learning_rate, grads, lr_scalers)
        return grads


class GradientClipping(LearningRule):
    """
    Clips gradients when they exceed a certain value.
    """
    def __init__(self, max_magnitude, rescale=False):
        self.max_magnitude = max_magnitude
        self.rescale = rescale

    @wraps(LearningRule.add_channels_to_monitor)
    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        pass

    @wraps(LearningRule.get_updates)
    def get_updates(self, learning_rate, grads, lr_scalers=None):
        updates = OrderedDict()
        if self.rescale:
            norm = 0
            for grad in grads.values():
                norm += (grad ** 2).sum()
            norm = T.switch(T.sqrt(norm) > self.max_magnitude,
                            self.max_magnitude / T.sqrt(norm), 1.)
            for param, grad in grads.items():
                updates[param] = param - learning_rate * (
                    lr_scalers.get(param, 1.) * grad * norm
                )
        else:
            for param, grad in grads.items():
                updates[param] = param - learning_rate * (
                    lr_scalers.get(param, 1.) * T.clip(grad,
                                                       -self.max_magnitude,
                                                       self.max_magnitude)
                )
        return updates
