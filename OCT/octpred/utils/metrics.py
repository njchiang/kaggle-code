import torch

# this will be a class that updates the loss values per epoch, etc.

class Metrics:
    def __init__(self, initial_value_dict, keep_every=1):
        self._initial = initial_value_dict
        self._total = initial_value_dict
        self._values = {k: [] for k in initial_value_dict}

    def update(self, value_dict):
        # TODO need error handling here 
        for k, v in value_dict.items():
            self._total[k] += v
            self._values[k].append(v)

    def reset(self):
        self._total = self._initial
        self._values = {k: [] for k in self._initial}

    def get_metrics(self):
        return self._total
    
    def get_history(self):
        return self._values
    