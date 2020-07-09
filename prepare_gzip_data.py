import gzip
import json
import argparse


parser = argparse.ArgumentParser(description='Turn json.gz data to json data.')
parser.add_argument('Path', metavar='path', type=str,
                    help='the path to input file')

args = parser.parse_args()

input_path = args.Path

with gzip.open(input_path, 'r') as f:
    data = json.loads(f.read(), encoding='utf-8')

with open('output.json', 'w') as f:
    json.dump(data, f)
