import yaml
from os import path


class ConfigLoader:
    name = '_config.yml'
    data = {}

    def __init__(self, root_abspath):
        self.root_abspath = root_abspath

        config_abspath = path.join(self.root_abspath, self.name)

        if path.exists(config_abspath):
            with open(config_abspath) as f:
                self.data = yaml.load(f)
