# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:18:58 2018

@author: boerbfde
"""
import numpy as np
import os
import pandas as pd

import cfg

def get_dict_eb_parse_meta():
    dict_eb_parse_meta = {}
    dict_eb_parse_meta['table'] = {}
    dict_eb_parse_meta['table']['tZ'] = {}
    dict_eb_parse_meta['table']['tY'] = {}
    dict_eb_parse_meta['table']['tRe'] = {}
    dict_eb_parse_meta['table']['tRm'] = {}
    dict_eb_parse_meta['table']['tRr'] = {}
    dict_eb_parse_meta['table']['tHe'] = {}
    dict_eb_parse_meta['table']['tHm'] = {}
    dict_eb_parse_meta['table']['tHr'] = {}
    dict_eb_parse_meta['table']['tW'] = {}

    dict_eb_parse_meta['table']['tZ']['file_name_pattern'] = 'mrIot'
    dict_eb_parse_meta['table']['tY']['file_name_pattern'] = 'mrFinalDemand'
    dict_eb_parse_meta['table']['tRe']['file_name_pattern'] = 'mrEmission'
    dict_eb_parse_meta['table']['tRm']['file_name_pattern'] = 'mrMaterial'
    dict_eb_parse_meta['table']['tRr']['file_name_pattern'] = 'mrResource'
    dict_eb_parse_meta['table']['tHe']['file_name_pattern'] = 'mrFDEmission'
    dict_eb_parse_meta['table']['tHm']['file_name_pattern'] = 'mrFDMaterial'
    dict_eb_parse_meta['table']['tHr']['file_name_pattern'] = 'mrFDResource'
    dict_eb_parse_meta['table']['tW']['file_name_pattern'] = 'mrFactorInput'

    dict_eb_parse_meta['table']['tZ']['index_col'] = [0, 1, 2]
    dict_eb_parse_meta['table']['tY']['index_col'] = [0, 1, 2]
    dict_eb_parse_meta['table']['tRe']['index_col'] = [0, 1, 2]
    dict_eb_parse_meta['table']['tRm']['index_col'] = [0, 1]
    dict_eb_parse_meta['table']['tRr']['index_col'] = [0, 1, 2]
    dict_eb_parse_meta['table']['tHe']['index_col'] = [0, 1, 2]
    dict_eb_parse_meta['table']['tHm']['index_col'] = [0, 1]
    dict_eb_parse_meta['table']['tHr']['index_col'] = [0, 1, 2]
    dict_eb_parse_meta['table']['tW']['index_col'] = [0, 1]

    dict_eb_parse_meta['table']['tZ']['header'] = [0, 1]
    dict_eb_parse_meta['table']['tY']['header'] = [0, 1]
    dict_eb_parse_meta['table']['tRe']['header'] = [0, 1]
    dict_eb_parse_meta['table']['tRm']['header'] = [0, 1]
    dict_eb_parse_meta['table']['tRr']['header'] = [0, 1]
    dict_eb_parse_meta['table']['tHe']['header'] = [0, 1]
    dict_eb_parse_meta['table']['tHm']['header'] = [0, 1]
    dict_eb_parse_meta['table']['tHr']['header'] = [0, 1]
    dict_eb_parse_meta['table']['tW']['header'] = [0, 1]

    return dict_eb_parse_meta


def fill_unit(df_source, df_target):
    '''

    '''
    list_df_source_column_values = list(df_source.columns.values)
    list_df_target_index_values = list(df_target.index.values)
    list_df_target_index_values_new = []
    for index_id, index in enumerate(list_df_target_index_values):
        unit = index[-1]
        if pd.isnull(unit):
            unit = list_df_source_column_values[index_id][-1]
        list_index = list(index)
        list_index[-1] = unit
        tuple_index = tuple(list_index)
        list_df_target_index_values_new.append(tuple_index)
    df_target.index = list_df_target_index_values_new
    return df_target


def parse():
    print('Begin parsing EXIOBASE')
    dict_eb_parse_meta = get_dict_eb_parse_meta()
    dict_eb_raw = {}

    # Get file names of exiobase.
    list_eb_file_name = os.listdir(cfg.eb_path)

    # Pattern match file names to fill dictionary with raw exiobase data.
    for eb_file_name in list_eb_file_name:
        for table in dict_eb_parse_meta['table']:
            if dict_eb_parse_meta['table'][table]['file_name_pattern'] in (
                    eb_file_name):
                eb_file_path = cfg.eb_path+eb_file_name
                dict_eb_raw[table] = pd.read_csv(
                        eb_file_path,
                        sep='\t',
                        header=dict_eb_parse_meta['table'][table]['header'],
                        index_col=dict_eb_parse_meta['table'][table]['index_col'],
                        low_memory=False)

    # Define file paths for characteristion factors.
    cQe_file_path = cfg.data_path+cfg.cQe_file_name
    cQm_file_path = cfg.data_path+cfg.cQm_file_name
    cQr_file_path = cfg.data_path+cfg.cQr_file_name

    # Read characterisation factors into pandas.
    df_cQe = pd.read_csv(cQe_file_path,
                         sep='\t',
                         header=[0, 1, 2],
                         index_col=[0, 1, 2, 3],
                         low_memory=False)
    df_cQm = pd.read_csv(cQm_file_path,
                         sep='\t',
                         header=[0, 1],
                         index_col=[0, 1],
                         low_memory=False)
    df_cQr = pd.read_csv(cQr_file_path,
                         sep='\t',
                         header=[0, 1, 2],
                         index_col=[0, 1],
                         low_memory=False)
    dict_eb_raw['cQe'] = df_cQe
    dict_eb_raw['cQm'] = df_cQm
    dict_eb_raw['cQr'] = df_cQr

    print('Done parsing EXIOBASE')
    return dict_eb_raw


def process(dict_eb_raw):
    print('Begin processing EXIOBASE')
    dict_eb_proc = {}

    # Construct Total Production Vector x from sum of Z and Y.
    df_tx = dict_eb_raw['tZ'].sum(axis=1) + dict_eb_raw['tY'].sum(axis=1)

    # Construct 1/x array for future calculations.
    array_tx = df_tx.values
    array_tx[array_tx == 0] = np.nan
    array_tx_inv = (1/array_tx)

    # Replace nan with zero, due to div by zero.
    array_tx_inv[np.isnan(array_tx_inv)] = 0

    # Construct Technical Coefficient Matrix.
    df_cA = dict_eb_raw['tZ']*array_tx_inv

    # Construct Leontief Inverse.
    array_cI = np.eye(df_cA.shape[0])
    array_cL = np.linalg.inv(array_cI-df_cA)
    df_cL = pd.DataFrame(array_cL,
                         index=df_cA.index,
                         columns=df_cA.columns)
    df_cL.index = df_cL.index.droplevel(2)
    df_cRe = dict_eb_raw['tRe']*array_tx_inv
    df_cRm = fill_unit(dict_eb_raw['cQe'], df_cRe)
    df_cRm = dict_eb_raw['tRm']*array_tx_inv
    df_cRm = fill_unit(dict_eb_raw['cQm'], df_cRm)
    df_cRr = dict_eb_raw['tRr']*array_tx_inv
    df_cRr = fill_unit(dict_eb_raw['cQr'], df_cRr)
    df_tY = dict_eb_raw['tY']
    df_tY.index = df_tY.index.droplevel(2)

    dict_eb_proc['cQe'] = dict_eb_raw['cQe']
    dict_eb_proc['cQm'] = dict_eb_raw['cQm']
    dict_eb_proc['cQr'] = dict_eb_raw['cQr']
    dict_eb_proc['cRe'] = df_cRe
    dict_eb_proc['cRm'] = df_cRm
    dict_eb_proc['cRr'] = df_cRr
    dict_eb_proc['cL'] = df_cL
    dict_eb_proc['tY'] = df_tY
    dict_eb_proc['tHe'] = dict_eb_raw['tHe']
    dict_eb_proc['tHm'] = dict_eb_raw['tHm']
    dict_eb_proc['tHr'] = dict_eb_raw['tHr']

    print('Done processing EXIOBASE')
    return dict_eb_proc


if __name__ == "__main__":

    dict_eb_proc = process(parse())

    df_cQRLe = dict_eb_proc['cQe'].dot(dict_eb_proc['cRe']).dot(
            dict_eb_proc['cL'])
    df_cQRLm = dict_eb_proc['cQm'].dot(dict_eb_proc['cRm']).dot(
            dict_eb_proc['cL'])
    df_cQRLr = dict_eb_proc['cQr'].dot(dict_eb_proc['cRr']).dot(
            dict_eb_proc['cL'])
