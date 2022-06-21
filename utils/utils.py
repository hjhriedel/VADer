import json
from dotmap import DotMap
import os
import time
import importlib
import argparse

def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return DotMap(config_dict), config_dict, json_file

def process_config(json_file):
    config, _, json_file = get_config_from_json(json_file)
    json_str = json_file.rsplit('_',1)[1].rsplit('.',1)[0] #e.g. mid-01
    dataloader_str =  'dl' + config.data_loader.name.rsplit('_',1)[1].rsplit('.',1)[0] #e.g. dl01
    model_str = 'm' + config.model.name.rsplit('_',1)[1].rsplit('.',1)[0] #e.g. m01
    config.callbacks.exp_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()))
    config.callbacks.checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()),"{}-{}-{}-checkpoints/".format(
          model_str,
          json_str,
          dataloader_str
          ))    
    return config

def create(cls):
    '''expects a string that can be imported as with a module.class name'''
    module_name, class_name = cls.rsplit(".",1)

    try:
        #print('importing '+module_name)
        somemodule = importlib.import_module(module_name)
        #print('getattr '+class_name)
        cls_instance = getattr(somemodule, class_name)
        #print(cls_instance)
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

    return cls_instance

def create_dirs(dirs):
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    return 0   

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-cd', '--config-dir',
        dest='config',
        metavar='C',
        default=r'configs',
        help='The Configuration file')

    argparser.add_argument(
        '-e', '--evaluate',
        dest='evaluate',
        metavar='C',
        default='false',
        help='Option to evaluate on test data')

    args = argparser.parse_args()
    return args
