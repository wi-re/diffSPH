# Wrapper class for parameters to allow for easier loading and storing of parameters in configs
# Also generally useful to simplify creation of new parameters by having a shared interface
class Parameter():
    def __init__(self, namespace, name, datatype, defaultValue, required = False, hint = '', export = True):
        self.namespace = namespace
        self.name = name
        self.datatype = datatype
        self.defaultValue = defaultValue
        self.required = required
        self.hint = hint
        self.export = export
        
    def __repr__(self):
        return 'Parameter [%s.%s], required? %s, exported? %s, description: %s' % (self.namespace, self.name, 'yes' if self.required else 'no', 'yes' if self.export else 'no', self.hint)

    def parseConfig(self, config):
        if self.namespace == '':
            if self.required and self.name not in config:
                raise Exception('Required parameter %s is missing' % self.name)
            elif self.name not in config:
                config[self.name] = self.defaultValue
            return

        if self.required and self.namespace not in config:
            raise Exception('Parameter Namespace %s missing for required parameter %s.%s' % (self.namespace, self.namespace, self.name))
        elif self.namespace not in config:
            config[self.namespace] = {}
                
        if self.required and self.name not in config[self.namespace]:
            raise Exception('Required parameter %s.%s is missing' % (self.namespace, self.name))
        elif self.name not in config[self.namespace]:
            config[self.namespace][self.name] = self.defaultValue
        
        
            