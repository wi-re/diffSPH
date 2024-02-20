import inspect
import diffSPH.v2.modules
class ModuleWrapper:
    def __init__(self, name, module):
        self.module = module
        self.name = name
    def __getattr__(self, name):
        return getattr(self.module, name)
    
    def getParameters(self):
        return self.module.getParameters() if hasattr(self.module,'getParameters') else []
    def getFunctions(self):
        return self.module.getFunctions() if hasattr(self.module,'getFunctions') else []
    

module_list = [(name, obj) for name, obj in inspect.getmembers(diffSPH.v2.modules) if inspect.ismodule(obj) and name != '__init__']
modules = []

for module_name, module in module_list:
    modules.append(ModuleWrapper(module_name, module))


