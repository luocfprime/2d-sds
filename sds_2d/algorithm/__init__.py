import importlib
import os
import pkgutil

excluded_modules = ["__init__"]

# automatically import all modules in the current directory to apply registry decorator
current_package = os.path.dirname(__file__)
for _, module_name, _ in pkgutil.iter_modules([current_package]):
    if module_name not in excluded_modules:
        importlib.import_module(f".{module_name}", __package__)
