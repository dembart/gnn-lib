import yaml



class Config(dict):
    def __init__(self, mapping=None):
        super().__init__()
        if mapping:
            for k, v in mapping.items():
                self[k] = self._convert(v)

    @classmethod
    def from_file(cls, yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def _convert(self, value):
        if isinstance(value, dict):
            return Config(value)
        elif isinstance(value, list):
            return [self._convert(v) for v in value]
        else:
            return value
    
    def to_yaml(self, path):
        def clean(obj):
            if isinstance(obj, Config):
                return {k: clean(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean(v) for v in obj]
            else:
                return obj
        
        with open(path, "w") as f:
            yaml.dump(clean(self), f, default_flow_style=False, sort_keys=False)


    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Missing config key: '{name}'")

    def __setattr__(self, name, value):
        self[name] = self._convert(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"Missing config key: '{name}'")



    


    