import importlib


def _factory(component_type, component_dict):
    comp_path, comp_module, comp_name = component_dict[component_type]
    module = importlib.import_module(comp_module, comp_path)
    return getattr(module, comp_name)
