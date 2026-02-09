import os
import importlib
import traceback

#Import RQP Function Files
path_PROJECT      = os.path.dirname(os.path.realpath(__file__))
path_rqpFunctions = os.path.join(path_PROJECT, 'rqpfunctions')
files_rqpfunctions = os.listdir(path_rqpFunctions)

#Search RQP Function Files and Import
RQPFUNCTIONS_MODEL                = dict()
RQPFUNCTIONS_INPUTDATAKEY         = dict()
RQPFUNCTIONS_BATCHPROCESSFUNCTION = dict()
for name_file in files_rqpfunctions:
    if not (name_file.startswith('rqpf_') and name_file.endswith('.py')): continue
    name_module   = name_file[:-3]
    name_function = name_file[5:-3]
    try:
        module = importlib.import_module(f"rqpfunctions.{name_module}")
        RQPFUNCTIONS_MODEL[name_function]                = getattr(module, 'MODEL')
        RQPFUNCTIONS_INPUTDATAKEY[name_function]         = getattr(module, 'INPUTDATAKEYS')
        RQPFUNCTIONS_BATCHPROCESSFUNCTION[name_function] = getattr(module, 'PROCESSBATCH')
    except Exception as e:
        traceback.print_exc()