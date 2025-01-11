import yaml

# Path to your scenarios.yaml
config_path = "C:/Users/leoco/AEPF_Mk3/config/scenarios.yaml"

# Load and print configuration
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(config)
