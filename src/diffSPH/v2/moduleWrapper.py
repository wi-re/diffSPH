import pathlib, importlib, re

class ModuleWrapper:
    def __init__(self, name, module):
        self.module = module
        self.name = name
        # for attr_name in dir(module):
        #     attr = getattr(module, attr_name)
        #     if callable(attr):
        #         setattr(self, attr_name, attr)
    def __getattr__(self, name):
        return getattr(self.module, name)
    
    def getParameters(self):
        return self.module.getParameters() if hasattr(self.module,'getParameters') else []
    def getFunctions(self):
        return self.module.getFunctions() if hasattr(self.module,'getFunctions') else []
    
import inspect
import diffSPH.v2.modules

module_list = [(name, obj) for name, obj in inspect.getmembers(diffSPH.v2.modules) if inspect.ismodule(obj) and name != '__init__']

# for module_name in module_list:
    # print(module_name)

modules = []

for module_name, module in module_list:
    modules.append(ModuleWrapper(module_name, module))
