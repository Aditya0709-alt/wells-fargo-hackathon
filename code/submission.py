#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import OrderedDict

class Assignment:
    def __init__(self,
                 assignment_name,
                 preds_full_range_dir='./data/preds_full_range/',
                 submission_file='./submission_init.txt',
                 verbose=True):
        
        self.assignment_name = assignment_name
        self.preds_full_range_dir = preds_full_range_dir
        self.submission_file = submission_file
        self.verbose = verbose
        
        self.assignment_name_column = None
        self.raw_submission = None
        
        self.preds_full_range = None
        self.preds_submission = None
        
    def load_raw_submission(self):
        """
        Load full submission file
        Suboptimal but it is a small file
        """
        raw_submission = pd.read_csv(self.submission_file,
                                      sep='\t',
                                      parse_dates=['DATE'], date_parser=np.datetime64)
        raw_submission = raw_submission.groupby('ASS_ASSIGNMENT')
        
        assignment = raw_submission.get_group(self.assignment_name)
        self.assignment_name_column = assignment['ASS_ASSIGNMENT'].reset_index(drop=True)
        
        self.raw_submission = assignment.set_index('DATE')
        return True
        
    def load_preds_full_range(self):
        """
        Load predictions made on the full range of dates
        """
        file_path = self.preds_full_range_dir + self.assignment_name
        file_path += '_predictions_full_range.csv'
        preds_full_range = pd.read_csv(file_path,
                                       sep=';',
                                       parse_dates=['DATE'], 
                                       index_col=['DATE'], date_parser=np.datetime64)
        self.preds_full_range = preds_full_range.drop('Value', 1)
        return True
    
    def set_preds_submission(self):
        """
        Retrieve only test predictions
        """
        self.preds_submission = \
        pd.concat([self.preds_full_range.loc[self.raw_submission.index].reset_index(),
                   self.assignment_name_column], axis=1)#.drop('index', 1)
        return True
    
    def process(self):
        self.load_raw_submission()
        self.load_preds_full_range()
        self.set_preds_submission()
        if self.verbose:
            print '[OK] Right predictions retrieved - %s'%self.assignment_name
        return True
        
class Submission:
    def __init__(self,
                 submission_file='./data/submission_init.txt',
                 preds_full_range_dir='./data/preds_full_range/',
                 output_file='./submissions.txt',
                 verbose=True):
        
        self.raw_submission = pd.read_csv(submission_file, sep='\t', parse_dates=['DATE'])
        self.assignment_names = self.raw_submission['ASS_ASSIGNMENT'].unique()
        
        self.preds_full_range_dir = preds_full_range_dir
        self.verbose = verbose
        self.output_file = output_file
        
        self.assignments = self.init_assignments()
        
        self.submission_final = self.raw_submission.copy()
    
    def init_assignments(self):
        assignments = OrderedDict()
        for assignment_name in sorted(self.assignment_names):
            assignment = Assignment(assignment_name,
                                    self.preds_full_range_dir,
                                    verbose=self.verbose)
            assignments[assignment_name] = assignment
        return assignments
        
    def process(self):
        for assignment_name, assignment in self.assignments.iteritems():
            assignment.process()
            
    def build_submission_final(self):
        res = []
        for assignment_name, assignment in self.assignments.iteritems():
            res.append(assignment.preds_submission)
        res = pd.concat(res, axis=0) \
        .set_index(['DATE', 'ASS_ASSIGNMENT']) \
        .sort_index() \
        .reset_index()
        self.submission_final = res
        self.submission_final['DATE'] = self.raw_submission['DATE']
        if self.verbose:
            print ''
            print '[OK] Submission dataframe built'
        return True
    
    def write_submission(self):
        self.submission_final.to_csv(path_or_buf=self.output_file,
                                     sep='\t',
                                     index=False,date_format='%Y-%m-%d %H:%M:00.000')
        if self.verbose:
            print ''
            print '[OK] Submission dataframe written in ' + self.output_file
        return True

if __name__ == "__main__":
    sub = Submission(output_file='./submissions.txt')
    sub.process()
    sub.build_submission_final()
    sub.write_submission()