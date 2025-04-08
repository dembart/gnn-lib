import torch.nn as nn
import yaml



def get_activation(activation_name):
    activation_dict = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softplus": nn.Softplus(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "none": None,  # No activation
    }
    if activation_name.lower() not in activation_dict:
        raise ValueError(f"Activation '{activation_name}' is not supported. "
                         f"Supported activations: {list(activation_dict.keys())}")

    return activation_dict[activation_name.lower()]



def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    config = ConfigDict(data)
    return config



class ConfigDict(dict):
    def __getattr__(self, name):
        if name in self:
            value = self[name]

            if isinstance(value, dict):
                return ConfigDict(value)
            
            return value
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")




    


    