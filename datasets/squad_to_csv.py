import argparse
import json

import pandas as pd
import numpy as np


def squad_json_to_dataframe(input_file_path,
                            record_path=['data','paragraphs','qas','answers'],
                            verbose=1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")    
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file , record_path )
    m = pd.io.json.json_normalize(file, record_path[:-1] )
    r = pd.io.json.json_normalize(file,record_path[:-2])
    
    #combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx  = np.repeat(m['id'].values,m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    main = pd.concat([ m[['id','question','context','answers']].set_index('id'),js.set_index('q_idx')],1,sort=False).reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main


parser = argparse.ArgumentParser(description='Turn SQuAD JSON to flat CSV')
parser.add_argument('--input', type=str,
                    help='the path to input file')
parser.add_argument('--output', type=str,
                    help='the path to output file')

args = parser.parse_args()

input_path = args.input
output_path = args.output

dataframe = squad_json_to_dataframe(input_path)

# Only use necessary columns and set appropriate names
dataframe = dataframe[['index', 'question', 'context', 'text', 'answer_start']]
dataframe.rename(columns={'text':'answer'}, inplace=True)

dataframe.to_csv(output_path)

