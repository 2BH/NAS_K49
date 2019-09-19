import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import numpy as np

class BOWorker(Worker):
  """"Worker for  for BOHB"""
  def __init__(self, train_func, **kwargs):
    """
    Initialization
    """
    super().__init__(**kwargs)
    self.train_func = train_func

  @staticmethod
  def get_configspace():
    """
    Construct the configuration space

    Return:
        cs: (ConfigurationSpace) A configuration space which defines the search space
    """
    cs = CS.ConfigurationSpace()
    num_layers = CSH.UniformIntegerHyperparameter("num_layers", lower=6, upper=10, default_value=8, log=False)
    model_learning_rate = CSH.UniformFloatHyperparameter('model_learning_rate', lower=2e-3, upper=3e-2, default_value=0.025, log=True)
    auxiliary_weight = CSH.UniformFloatHyperparameter('auxiliary_weight', lower=0.2, upper=0.5, default_value=0.4, log=False)
    init_channel = CSH.UniformIntegerHyperparameter('init_channel', lower=12, upper=24, default_value=16, log=False)
    drop_path_prob = CSH.UniformFloatHyperparameter('drop_path_prob', lower=0.2, upper=0.4, default_value=0.3, log=False)
    weight_decay = CSH.UniformFloatHyperparameter("weight_decay", lower=1e-4, upper=2e-3, default_value=3e-4, log=True)
    optimizer = CSH.CategoricalHyperparameter(name='optimizer', choices=['Adam', 'AdamW'])

    cs.add_hyperparameters([num_layers, model_learning_rate, auxiliary_weight,
                            init_channel, drop_path_prob, weight_decay, optimizer])

    return cs

  def compute(self, config, budget, **kwargs):
    """
    Evaluates the configuration on the defined budget and returns the validation performance.
    Args:
        config: dictionary containing the sampled configurations by the optimizer
        budget: (float) amount of time/epochs/etc. the model can use to train
    Returns:
        dictionary with mandatory fields:
            'loss' (scalar)
            'info' (dict)
    """
    # Run one configuration
    valid_acc_bal = self.train_func(config, int(budget))

    return ({
      'loss': 1 - valid_acc_bal,
      'info': {
      "valid_accuracy_balanced": valid_acc_bal
      }
    })
