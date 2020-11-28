import pandas as pd
import numpy as np

class raw_data:
    
    def __init__(self, fp, ind1, ind2): 
        self.fp = fp     # filepath 
        self.ind1 = ind1 # column index with the biased word
        self.ind2 = ind2 # column index with the revision
    
    def read_data(self): 
        ''' reads in dataset and returns as numpy matrix'''
        return np.array(pd.read_table(self.fp))

    def find_missing_words(self, s1, s2):
        '''Given sentences s1 and s2, finding the words emitted from s1 to s2'''
        s1_list = s1.split()
        s2_list = s2.split()

        diff = [word for word in s1_list + s2_list if word not in s1_list or word not in s2_list]
        return diff

    def find_emitted_words(self, data):
        '''find emitted words in each row, and return a numpy array of all revised words per sentence.'''
        diff_words = list()
        for sentence in data: 
            diff_words.append(self.find_missing_words(sentence[self.ind1], sentence[self.ind2]))
        return np.array([np.array(row) for row in diff_words])

    def add_miss_word_col(self, dtype='np', cols=None):
        '''
        Given a filepath and column number of the sentence data, it will
        return a combined dataset with the missing words as an additional column
        '''
        # get missing words as numpy array
        mis_words = self.find_emitted_words(self.read_data())
        # combine the datasets
        combined_data = np.column_stack((self.read_data(), mis_words))
        
        if dtype == 'np':
            return combined_data
        if dtype =='df':
            return pd.DataFrame(combined_data, columns=cols)
        else: 
            raise ValueError('Wrong data type. Only numpy arrays and dataframe are accepted as input.')