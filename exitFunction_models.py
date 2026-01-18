import os
import importlib
import pprint

#Import RQP Function Files
path_PROJECT      = os.path.dirname(os.path.realpath(__file__))
path_rqpFunctions = os.path.join(path_PROJECT, 'rqpfunctions')
files_rqpfunctions = os.listdir(path_rqpFunctions)

#Search RQP Function Files and Import
RQPFUNCTIONS_MODEL = dict()
for name_file in files_rqpfunctions:
    if name_file.startswith('rqpf_') and name_file.endswith('.py'):
        name_module   = name_file[:-3]
        name_function = name_file[5:-3]
        try:
            module = importlib.import_module(f"rqpfunctions.{name_module}")
            model  = getattr(module, 'MODEL')
            RQPFUNCTIONS_MODEL[name_function] = model
        except Exception as e:
            print(e)