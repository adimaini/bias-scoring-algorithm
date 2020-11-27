import pandas as pd
import numpy as np

def data(train_fp, test_fp): 
    ''' reads data and returns test and train datasets in numpy arrays'''
    cols = ['id', 'src_tok', 'tgt_tok', 'src_raw', 'tgt_raw', 'src_POS_tags', 'tgt_parse_tags']
    df_train = pd.read_table(train_fp, names=cols)
    df_test = pd.read_table(test_fp, names=cols)

    train = np.array(df_train)
    test = np.array(df_test)

    return train, test