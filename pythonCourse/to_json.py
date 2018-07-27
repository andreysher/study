import json


def to_json(func):
    def decorated(*args, **kwargs):
        return json.dumps(func(*args, **kwargs))

    return decorated


@to_json
def my_func():
    return {'data': 42}


print(my_func())
