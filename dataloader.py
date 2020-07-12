import nlp
import numpy as np


if __name__ == '__main__':
    mc_test = nlp.load_dataset('./datasets/mc_test.py')
    print('MCTest train data: %s' % len(mc_test['train']))
    print('MCTest test data: %s' % len(mc_test['test']))

    print('-'*30)
    news_qa = nlp.load_dataset('./datasets/news_qa.py')
    print('NewsQA train data: %s' % len(news_qa['train']))
    print('NewsQA test data: %s' % len(news_qa['test']))

    print('-'*30)
    squad = nlp.load_dataset('squad')
    print('SQuAD train data: %s' % len(squad['train']))
    print('SQuAD val data: %s' % len(squad['validation']))

    print('-'*30)
    cosmos = nlp.load_dataset('cosmos_qa')
    print('Cosmos QA train data: %s' % len(cosmos['train']))
    print('Cosmos QA val data: %s' % len(cosmos['validation']))
    print('Cosmos QA test data: %s' % len(cosmos['test']))
