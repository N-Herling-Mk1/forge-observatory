"""Model registry for the genre project. train.py looks models up by config key."""
REGISTRY = {}
def register(name):
    def deco(fn): REGISTRY[name] = fn; return fn
    return deco
