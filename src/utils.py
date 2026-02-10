import yaml
import os

def load_config(config_path):
    #load YAML config file
    with open(config_path,'r') as f:
        return yaml.safe_load(f)

def save_config(config_dict,config_path):
    #save YAML config file
    with open(config_path,'w') as f:
        yaml.dump(config_dict,f)

def ensure_dir_exists(directory):
    #Ensure directory exists
    os.makedirs(directory,exist_ok=True)