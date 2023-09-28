import yaml
from flex_env import FlexEnv

def load_yaml(filename):
    # load YAML file
    return yaml.safe_load(open(filename, 'r'))

# load config
config = load_yaml("config/data_gen/gnn_dyn.yaml")

env = FlexEnv(config)
env.reset()