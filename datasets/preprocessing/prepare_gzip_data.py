import gzip
import json
import argparse


parser = argparse.ArgumentParser(description='Turn json.gz data to json data.')
parser.add_argument('--input', type=str,
                    help='the path to input file')
parser.add_argument('--output', type=str,
                    help='the path to output file')

args = parser.parse_args()

input_path = args.input
output_path = args.output

with gzip.open(input_path, 'r') as f:
    data = json.loads(f.read(), encoding='utf-8')

with open(output_path, 'w') as f:
    json.dump(data, f)
