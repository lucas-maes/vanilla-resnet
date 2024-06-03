import importlib
import inspect
import json

from dataclasses import dataclass, field, is_dataclass

def load_dataclass(config_path):
    with open(config_path, 'r') as file:
        file_content = file.read()

    global_context = {
        'dataclass': dataclass,
        'field': field,
        'is_dataclass': is_dataclass
    }
    local_scope = {}

    exec(file_content, global_context, local_scope)

    for name, obj in local_scope.items():
        if inspect.isclass(obj) and is_dataclass(obj):
            return obj

    raise ValueError("No dataclass found in the file.")


def dump_json(config_dataclass):
    json_dict = {}
    for field in config_dataclass.__dataclass_fields__.keys():
        json_dict[field] = getattr(config_dataclass, field)

    return json.dumps(json_dict)

def load_json(json_str, config_dataclass):
    json_dict = json.loads(json_str)
    return config_dataclass(**json_dict)
