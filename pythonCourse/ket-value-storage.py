import argparse
import json
import tempfile
import os

parser = argparse.ArgumentParser()
parser.add_argument('--key', help='''if you use --value, store key:value in storage,
 else give you value for this key''', type=str)
parser.add_argument('--value', help='''use with --key''', type=str)

args = parser.parse_args()

storage_path = os.path.join(tempfile.gettempdir(), 'storage.data')

if args.key and args.value:
    with open(storage_path, 'r') as f:
        json_decoded = json.load(f)
        try:
            json_decoded[args.key] += ', ' + args.value
        except KeyError:
            json_decoded[args.key] = args.value

    with open(storage_path, 'w') as f:
        json.dump(json_decoded, f)
        print(json_decoded)
    exit(0)

if args.key:
    with open(storage_path, 'r') as f:
        json_decoded = json.load(f)
        try:
            print(json_decoded[args.key])
        except KeyError:
            print('')