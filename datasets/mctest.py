import pandas as pd
import numpy as np


def convert_mctest_to_csv(X_df, y_df):
    contexts = np.repeat(X_df[2].values, 4)
    questions = X_df[[3, 8, 13, 18]].copy().to_numpy().flatten()

    replacer = lambda x: x.replace('multiple: ', '').replace('one: ', '')
    questions = np.array([replacer(q) for q in questions])

    answers = X_df[[4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22]].copy()
    answers = answers.to_numpy().reshape(len(questions), 4)

    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    true_answer_codes = y_df.to_numpy().flatten()

    true_answers = []
    for i,answer in enumerate(answers):
        true_answers.append(answer[mapping[true_answer_codes[i]]])

    true_answers = np.array(true_answers)

    output = pd.DataFrame()

    output['context'] = contexts
    output['question'] = questions
    output['answer'] = true_answers

    return output


if __name__ == '__main__':
    X_dev_df = pd.read_csv('mctest/raw/mc500.dev.tsv', sep='\t', header=None)
    y_dev_df = pd.read_csv('mctest/raw/mc500.dev.ans', sep='\t', header=None)

    X_train_df = pd.read_csv('mctest/raw/mc500.train.tsv', sep='\t', header=None)
    y_train_df = pd.read_csv('mctest/raw/mc500.train.ans', sep='\t', header=None)

    X_test_df = pd.read_csv('mctest/raw/mc500.test.tsv', sep='\t', header=None)
    y_test_df = pd.read_csv('mctest/raw/mc500.test.ans', sep='\t', header=None)

    X_train_df = pd.concat([X_dev_df, X_train_df])
    y_train_df = pd.concat([y_dev_df, y_train_df])

    train_output = convert_mctest_to_csv(X_train_df, y_train_df)
    train_output.to_csv('mctest/mctest_train.csv')

    test_output = convert_mctest_to_csv(X_test_df, y_test_df)
    test_output.to_csv('mctest/mctest_test.csv')

