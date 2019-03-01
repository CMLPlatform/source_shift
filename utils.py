# -*- coding: utf-8 -*-
""" Utilities for script of paper on
    potential reductions in the environmental footprints embodied in
    European Union’s imports through source shifting
    Copyright (C) 2018

    Bertram F. de Boer
    Faculty of Science
    Institute of Environmental Sciences (CML)
    Department of Industrial Ecology
    Einsteinweg 2
    2333 CC Leiden
    The Netherlands

    +31 (0)71 527 1478
    b.f.de.boer@cml.leidenuniv.nl

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

from collections import OrderedDict

import csv
import math
import matplotlib.collections as mpl_col
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pandas as pd
import pickle

import cfg
import exiobase as eb


def cm2inch(tup_cm):
    """ Convert cm to inch.
        Used for figure generation.

        Parameters
        ----------
        tup_cm: tuple with values in cm.

        Returns
        -------
        tup_inch: tuple with values in inch.

    """
    inch = 2.54
    tup_inch = tuple(i/inch for i in tup_cm)
    return tup_inch


def get_df_tY_eu28(dict_eb):
    """ Get final demand matrix of EU28.

        Parameters:
        -----------
        dict_eb: dictionary with parsed version of EXIOBASE.

        Returns:
        --------
        df_tY_eu28: DataFrame with final demand of EU28.

    """
    list_reg_fd = get_list_reg_fd('EU28')
    df_tY_eu28 = dict_eb['tY'][list_reg_fd].copy()
    for cntr in list_reg_fd:
        df_tY_eu28.loc[cntr, cntr] = 0
    return df_tY_eu28


def get_dict_df_imp(dict_cf, dict_eb, df_tY):
    """ Calculate footprints given final demand

        Parameters:
        ----------
        dict_cf: Dictionary with characterization factors for footprints
        dict_eb: Dictionary with processed version of EXIOBASE.
        df_tY: DataFrame with final demand

        Returns:
        --------
        dict_df_imp: dictionary with DataFrames of footprints

    """
    #   Diagonalize final demand.
    ar_tY = np.diag(df_tY)
    df_tYd = pd.DataFrame(ar_tY, index=df_tY.index, columns=df_tY.index)

    #   Calculate absolute impact of imported products to EU28.
    df_cQe = dict_cf['e']
    df_cQm = dict_cf['m']
    df_cQr = dict_cf['r']

    df_cRe = dict_eb['cRe']
    df_cRm = dict_eb['cRm']
    df_cRr = dict_eb['cRr']

    df_cL = dict_eb['cL']

    dict_df_imp = {}
    dict_df_imp['e'] = df_cQe.dot(df_cRe).dot(df_cL).dot(df_tYd)
    dict_df_imp['m'] = df_cQm.dot(df_cRm).dot(df_cL).dot(df_tYd)
    dict_df_imp['r'] = df_cQr.dot(df_cRr).dot(df_cL).dot(df_tYd)

    return dict_df_imp


def get_dict_imp(dict_df_imp):
    """ Convert dictionary with DataFrames of footprints to
        dictionary with footprints

        Parameters:
        ----------
        dict_df_imp: dictionary with DataFrames of footprints

        Returns:
        --------
        dict_imp: dictionary with footprints

    """

    dict_imp = {}
    for cat in dict_df_imp:
        dict_df = dict_df_imp[cat].T.to_dict()
        for imp_cat in dict_df:
            dict_imp[imp_cat] = {}
            for tup_prod_cntr in dict_df[imp_cat]:
                cntr, prod = tup_prod_cntr
                if prod not in dict_imp[imp_cat]:
                    dict_imp[imp_cat][prod] = {}
                dict_imp[imp_cat][prod][cntr] = dict_df[imp_cat][tup_prod_cntr]
    return dict_imp


def get_dict_imp_cat_unit():
    """ Generate dictionary to convert orders of magnitude.
        Used for plotting.

        Returns:
        --------
        dict_imp_cat_unit: dictionary linking orders of magnitude for each
                           footprint.
    """
    dict_imp_cat_unit = {}
    dict_imp_cat_unit['kg CO2 eq.'] = r'$Pg\/CO_2\/eq.$'
    dict_imp_cat_unit['kt'] = r'$Gt$'
    dict_imp_cat_unit['Mm3'] = r'$Mm^3$'
    dict_imp_cat_unit['km2'] = r'$Gm^2$'
    return dict_imp_cat_unit


def get_cf(file_path, df_cQ):
    """ Extract characterization factors of footprints from DataFrame

        Parameters:
        -----------
        file_path: string with path to file containing names of footprints
        df_cQ: DataFrame with characterization factors

    """
    list_imp = []
    with open(file_path) as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_imp.append(tuple(row))
    return df_cQ.loc[list_imp]


def get_dict_cf(dict_eb):
    """ Generate dictionary with characterization factors for footprints

        Parameters:
        dict_eb: dictionary with processed version of EXIOBASE.

        Returns:
        dict_cf: dictionary with DataFrames of characterization factors.

    """
    dict_cf = {}
    dict_cf['e'] = get_cf(cfg.data_path+cfg.e_fp_file_name,
                          dict_eb['cQe'])
    dict_cf['m'] = get_cf(cfg.data_path+cfg.m_fp_file_name,
                          dict_eb['cQm'])
    dict_cf['r'] = get_cf(cfg.data_path+cfg.r_fp_file_name,
                          dict_eb['cQr'])
    return dict_cf


def get_dict_eb():
    """ Load EXIOBASE.

        Returns:
        --------
        dict_eb: dictionary with processed version of EXIOBASE.

    """
    # If EXIOBASE has already been parsed, read the pickle.
    if cfg.dict_eb_file_name in os.listdir(cfg.data_path):
        dict_eb = pickle.load(open(cfg.data_path+cfg.dict_eb_file_name, 'rb'))
    # Else, parse and process EXIOBASE and optionally save for future runs
    else:
        dict_eb = eb.process(eb.parse())
        if cfg.save_eb:
            pickle.dump(dict_eb, open(cfg.data_path+cfg.dict_eb_file_name,
                                      'wb'))
    return dict_eb


def get_dict_prod_long_short():
    """ Generate dictionary linking long and short versions of product names.
        Used for plotting.

        Returns:
        --------
        dict_prod_long_short: dictionary linking long and short versions of
                              product names.

    """

    list_prod_long = []
    with open(cfg.data_path+cfg.prod_long_file_name) as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_prod_long.append(row[0])

    list_prod_short = []
    with open(cfg.data_path+cfg.prod_short_file_name) as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_prod_short.append(row[0])

    dict_prod_long_short = {}
    for prod_id, prod_long in enumerate(list_prod_long):
        prod_long = list_prod_long[prod_id]
        prod_short = list_prod_short[prod_id]
        dict_prod_long_short[prod_long] = prod_short
    return dict_prod_long_short


def get_dict_cntr_short_long():
    """ Generate dictionary linking codes and names of countries and regions.
        Used for plotting.

        Returns:
        --------
        dict_cntr_short_long: dictionary linking codes and names of
                              countries and regions.

    """
    dict_cntr_short_long = {}
    with open(cfg.data_path+cfg.country_code_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            cntr_short = row[0]
            cntr_long = row[1]
            dict_cntr_short_long[cntr_short] = cntr_long
    return dict_cntr_short_long


def get_dict_imp_cat_fp():
    """ Generate dictionary linking impact categories and footprint names.
        Used for plotting.

        Returns:
        --------
        dict_imp_cat_fp: dictionary linking
                         impact categories and footprint names.

    """
    dict_imp_cat_fp = {}
    with open(cfg.data_path+cfg.cf_long_footprint_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            imp_cat = tuple(row[:-1])
            fp = row[-1]
            dict_imp_cat_fp[imp_cat] = fp
    return dict_imp_cat_fp


def get_dict_imp_cat_magnitude():
    """ Generate dictionary linking footprints with order of magnitudes.
        Used for plotting.

        Returns:
        --------
        dict_imp_cat_magnitude: dictionary linking
                                footprints with order of magnitudes

    """
    dict_imp_cat_magnitude = {}
    with open(cfg.data_path+cfg.cf_magnitude_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            imp_cat = tuple(row[:-1])
            magnitude = int(row[-1])
            dict_imp_cat_magnitude[imp_cat] = magnitude
    return dict_imp_cat_magnitude


def get_list_prod_order_cons():
    """ Generate concatinated and ordered list of highest contributing products
        for each footprint.
        Used for plotting.

        Returns:
        --------
        list_prod_order_cons: concatinated and ordered list of
                              highest contributing products for each footprint.

    """
    list_prod_order_cons = []
    with open(cfg.data_path+cfg.prod_order_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_prod_order_cons.append(row[0])
    list_prod_order_cons.reverse()
    return list_prod_order_cons


def get_list_reg_fd(reg_fd):
    """ Generate list with country codes for all EU28 countries.
        Used to select final demand matrix.

        Returns:
        --------
        list_reg_fd: list with country codes for all EU28 countries.

    """

    if reg_fd == 'EU28':
        with open(cfg.data_path+cfg.eu28_file_name) as read_file:
            csv_file = csv.reader(read_file, delimiter='\t')
            list_reg_fd = []
            for row in csv_file:
                list_reg_fd.append(row[0])
    else:
        list_reg_fd = [reg_fd]
    return list_reg_fd


def makedirs(result_dir_name):
    """ Make directories for results.

    """
    print('\nMaking output directories in:\n\t{}'.format(
            cfg.result_dir_path+result_dir_name))
    for output_dir_name in cfg.list_output_dir_name:
        try:
            output_dir_path = (cfg.result_dir_path
                               + result_dir_name
                               + output_dir_name)
            os.makedirs(output_dir_path)
        except FileExistsError as e:
            print('\n\tOutput directory already exists:\n'
                  '\t{}\n'
                  '\tThis run will overwrite previous output.'.format(
                          output_dir_path))


class Priority:
    """ This class is used to calculate highest contributing products to each
    footprint.

    """

    dict_imp_cat_fp = get_dict_imp_cat_fp()
    dict_prod_long_short = get_dict_prod_long_short()
    dict_result = OrderedDict()

    def __init__(self):
        print('\nCreating instance of Priority class.')
        makedirs(cfg.priority_setting_dir_name)

    def calc(self, dict_cf, dict_eb, df_tY_eu28):
        """ Calculate highest contributing products to each footprint.

        Parameters
        ----------
        dict_cf: dictionary with characterisation factors of footprints
        dict_eb: dictionary with processed version of EXIOBASE
        df_tY_eu28: DataFrame with EU28 imported final demand

        """
        print('\nCalculating highest contributing products to each footprint.')

        # Sum over all EU28 countries and imported final demand categories.
        df_tY_eu28_fdsum = df_tY_eu28.sum(axis=1)

        # Calculate EU28 import embodied footprints.
        dict_df_imp = get_dict_df_imp(dict_cf, dict_eb, df_tY_eu28_fdsum)

        # Select and order highest contributing products up to configured limit
        for cat in dict_df_imp:
            df_imp_prod = dict_df_imp[cat].sum(axis=1, level=1)
            dict_imp_prod = df_imp_prod.T.to_dict()
            dict_imp_prod_sum = dict_df_imp[cat].sum(axis=1).to_dict()
            for imp_cat in dict_imp_prod:
                self.dict_result[imp_cat] = OrderedDict()
                list_imp_sort = sorted(dict_imp_prod[imp_cat].items(),
                                       key=operator.itemgetter(1),
                                       reverse=True)
                imp_cum = 0
                bool_add = True
                for tup_prod_abs_id, tup_prod_abs in enumerate(list_imp_sort):
                    (prod, imp_abs) = tup_prod_abs
                    imp_rel = imp_abs/dict_imp_prod_sum[imp_cat]
                    imp_cum = imp_cum + imp_rel
                    if imp_cum < cfg.imp_cum_lim_priority:
                        self.dict_result[imp_cat][prod] = imp_abs
                    elif bool_add:
                        self.dict_result[imp_cat][prod] = imp_abs
                        bool_add = False

    def log(self):
        """ Write highest contributing products for each footprint to file

        """
        priority_setting_dir_path = (cfg.result_dir_path
                                     + cfg.priority_setting_dir_name)

        log_file_name = 'log.txt'
        log_file_path = (priority_setting_dir_path
                         + cfg.txt_dir_name
                         + log_file_name)
        print('\nSaving priority setting log to:\n\t{}'.format(
                priority_setting_dir_path+cfg.txt_dir_name))
        with open(log_file_path, 'w') as write_file:
            csv_file = csv.writer(write_file,
                                  delimiter='\t',
                                  lineterminator='\n')
            csv_file.writerow(['Footprint', 'Product'])
            for imp_cat in self.dict_result:
                csv_file.writerow([])
                fp = self.dict_imp_cat_fp[imp_cat]
                for prod in self.dict_result[imp_cat]:
                    csv_file.writerow([fp, prod])

    def plot(self):
        """ Plot highest contributing products of each footprint.


        """

        dict_imp_cat_magnitude = get_dict_imp_cat_magnitude()

        analysis_name = 'priority_setting'
        priority_setting_dir_path = (cfg.result_dir_path
                                     + cfg.priority_setting_dir_name)
        pdf_dir_path = priority_setting_dir_path+cfg.pdf_dir_name
        png_dir_path = priority_setting_dir_path+cfg.png_dir_name

        print('\nSaving priority setting plots to:\n\t{}\n\t{}'.format(
                pdf_dir_path,
                png_dir_path))

        plt.close('all')
        dict_imp_cat_unit = get_dict_imp_cat_unit()
        fig = plt.figure(figsize=cm2inch((16, cfg.font_size)), dpi=cfg.dpi)
        for imp_cat_id, imp_cat in enumerate(self.dict_result):
            plot_id = imp_cat_id+1
            plot_loc = 220+plot_id
            ax = fig.add_subplot(plot_loc)
            fp = self.dict_imp_cat_fp[imp_cat]
            unit = dict_imp_cat_unit[imp_cat[-1]]
            ax.set_xlabel('{} [{}]'.format(fp, unit))
            df = pd.DataFrame(self.dict_result[imp_cat],
                              index=['import'])
            df.rename(columns=self.dict_prod_long_short, inplace=True)
            df_column_order = list(df.columns)
            df_column_order.reverse()
            df = df.reindex(df_column_order, axis=1)
            column_name_dummy = ''
            prod_order_dummy = df_column_order
            while len(df.T) < 9:
                df[column_name_dummy] = 0
                prod_order_dummy.reverse()
                prod_order_dummy.append(column_name_dummy)
                prod_order_dummy.reverse()
                df = df.reindex(df_column_order, axis=1)
                column_name_dummy += ' '

            df.T.plot.barh(stacked=True,
                           ax=ax,
                           legend=False,
                           color='C0',
                           width=0.8)

            yticklabels = ax.get_yticklabels()
            ax.set_yticklabels(yticklabels)

            plt.locator_params(axis='x', nbins=1)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

            xlim = ax.get_xlim()
            xlim_max_magn = 10**np.floor(np.log10(xlim[1]))
            xlim_max_ceil = math.ceil(xlim[1]/xlim_max_magn)*xlim_max_magn
            tup_xlim_max_ceil = (int(xlim[0]), xlim_max_ceil)

            ax.set_xlim(tup_xlim_max_ceil)
            xtick_magnitude = dict_imp_cat_magnitude[imp_cat]

            list_xtick = [i/xtick_magnitude for i in tup_xlim_max_ceil]
            list_xtick[0] = int(list_xtick[0])
            ax.set_xticks(list(tup_xlim_max_ceil))
            ax.set_xticklabels(list_xtick)

            xtick_objects = ax.xaxis.get_major_ticks()
            xtick_objects[0].label1.set_horizontalalignment('left')
            xtick_objects[-1].label1.set_horizontalalignment('right')
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.yaxis.set_tick_params(size=0)
        fig.tight_layout()
        plt.subplots_adjust(wspace=1)

        fig_file_name = analysis_name+'.pdf'
        fig_file_path = pdf_dir_path+fig_file_name
        fig.savefig(fig_file_path)

        fig_file_name = analysis_name+'.png'
        fig_file_path = png_dir_path+fig_file_name
        fig.savefig(fig_file_path)


class SourceShift():
    """ This class is used to calculate reductions in
        import embedded footprints by source shifting.

    """

    dict_imp_cat_fp = get_dict_imp_cat_fp()
    dict_imp_cat_magnitude = get_dict_imp_cat_magnitude()
    dict_prod_long_short = get_dict_prod_long_short()
    dict_shift_result = {}
    dict_reduc_result = {}

    def __init__(self):
        print('\nCreating instance of SourceShift class.')
        makedirs(cfg.shift_dir_name)
        makedirs(cfg.reduc_dir_name)

    def calc_shift(self, dict_cf, dict_eb, df_tY_eu28):
        """ Optimize sourcing for each footprint.

        Parameters
        ----------
        dict_cf: dictionary with characterisation factors of footprints
        dict_eb: dictionary with processed version of EXIOBASE
        df_tY_eu28: DataFrame with EU28 imported final demand

        """
        print('\nReducing import embodied footprints of EU28 by '
              'source shifting.')

        # Calculate import embodied footprints.
        df_tY_eu28_fdsum = df_tY_eu28.sum(axis=1)
        dict_df_imp = get_dict_df_imp(dict_cf, dict_eb, df_tY_eu28_fdsum)
        dict_imp = get_dict_imp(dict_df_imp)

        # Calculate footprint intensities per M Euro
        dict_df_imp_pME = {}
        dict_df_imp_pME['e'] = dict_cf['e'].dot(dict_eb['cRe']).dot(
                dict_eb['cL'])
        dict_df_imp_pME['m'] = dict_cf['m'].dot(dict_eb['cRm']).dot(
                dict_eb['cL'])
        dict_df_imp_pME['r'] = dict_cf['r'].dot(dict_eb['cRr']).dot(
                dict_eb['cL'])
        dict_imp_pME = get_dict_imp(dict_df_imp_pME)

        # For each footprint, for each product, sort countries according to
        # footprint intensity.
        dict_imp_prod_cntr_sort = {}
        for imp_cat in dict_imp:
            dict_imp_prod_cntr_sort[imp_cat] = OrderedDict()
            for prod in dict_imp[imp_cat]:
                dict_imp_prod_cntr_sort[imp_cat][prod] = OrderedDict()
                list_imp_pME_prod_cntr_sort = sorted(
                            dict_imp_pME[imp_cat][prod].items(),
                            key=operator.itemgetter(1))
                for tup_cntr_imp_pME in list_imp_pME_prod_cntr_sort:
                    cntr, imp_pME_prod_cntr = tup_cntr_imp_pME
                    dict_imp_prod_cntr_sort[imp_cat][prod][cntr] = (
                            dict_imp[imp_cat][prod][cntr])

        # For all products, for all countries,
        # calculate how much is imported by EU28.
        df_tY_eu28_cntr = df_tY_eu28.sum(axis=1, level=0)
        dict_tY_eu28_cntr = df_tY_eu28_cntr.to_dict()
        dict_tY_eu28_cntr_import = {}
        for cntr_fd in dict_tY_eu28_cntr:
            for tup_cntr_prod in dict_tY_eu28_cntr[cntr_fd]:
                cntr, prod = tup_cntr_prod
                if prod not in dict_tY_eu28_cntr_import:
                    dict_tY_eu28_cntr_import[prod] = {}

                if cntr not in dict_tY_eu28_cntr_import[prod]:
                    dict_tY_eu28_cntr_import[prod][cntr] = 0

                if cntr_fd is not cntr:
                    dict_tY_eu28_cntr_import[prod][cntr] += (
                            dict_tY_eu28_cntr[cntr_fd][tup_cntr_prod])

        # For all products, for all countries,
        # calculate how much is exported in total.
        df_tY_world = dict_eb['tY'].copy()
        list_cntr = list(df_tY_world.columns.get_level_values(0))
        for cntr in list_cntr:
            df_tY_world.loc[cntr, cntr] = 0
        df_tY_world_ex = df_tY_world.sum(axis=1)
        dict_tY_world_ex = df_tY_world_ex.to_dict()
        dict_tY_world_ex_prod_cntr = {}
        for tup_cntr_prod in dict_tY_world_ex:
            cntr, prod = tup_cntr_prod
            if prod not in dict_tY_world_ex_prod_cntr:
                dict_tY_world_ex_prod_cntr[prod] = {}
            dict_tY_world_ex_prod_cntr[prod][cntr] = (
                    dict_tY_world_ex[tup_cntr_prod])

        # Initialize dictionary with results.
        for imp_cat in dict_imp_prod_cntr_sort:
            self.dict_shift_result[imp_cat] = {}
            for prod in dict_imp_prod_cntr_sort[imp_cat]:
                self.dict_shift_result[imp_cat][prod] = {}
                for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
                    imp_pME_prod_cntr = dict_imp_pME[imp_cat][prod][cntr]
                    y_prod_cntr = dict_tY_eu28_cntr_import[prod][cntr]
                    x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
                    if x_prod_cntr >= cfg.x_prod_cntr_min:
                        self.dict_shift_result[imp_cat][prod][cntr] = {}
                        self.dict_shift_result[imp_cat][prod][cntr]['imp_pME'] = (
                                imp_pME_prod_cntr)
                        self.dict_shift_result[imp_cat][prod][cntr]['export'] = (
                                x_prod_cntr)
                        self.dict_shift_result[imp_cat][prod][cntr]['EU_import_old'] = (
                                y_prod_cntr)
                        self.dict_shift_result[imp_cat][prod][cntr]['EU_import_new'] = (
                                                        0)

        # For each product, calculate total EU28 imports.
        dict_tY_prod = {}
        for prod in dict_tY_eu28_cntr_import:
            dict_tY_prod[prod] = 0
            for cntr in dict_tY_eu28_cntr_import[prod]:
                x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
                if x_prod_cntr >= cfg.x_prod_cntr_min:
                    dict_tY_prod[prod] += dict_tY_eu28_cntr_import[prod][cntr]

        # For each footprint, for each product, for each exporting country
        # Shift imports from EU28 to countries with lowest impact intensity
        # Up to current level of country export
        for imp_cat in dict_imp_prod_cntr_sort:
            for prod in dict_imp_prod_cntr_sort[imp_cat]:
                y_prod = dict_tY_prod[prod]
                for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
                    x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
                    # Exclude countries with very low exports, due to noise.
                    if x_prod_cntr >= cfg.x_prod_cntr_min:
                        # If export of country is smaller than remaining
                        # EU28 import, redirect all exports to EU 28
                        if x_prod_cntr < y_prod:
                            self.dict_shift_result[imp_cat][prod][cntr]['EU_import_new'] = (
                                                            x_prod_cntr)
                            y_prod -= x_prod_cntr
                        # Else, redirect exports to EU28 up to remaining level
                        # of import.
                        elif y_prod > 0:
                            self.dict_shift_result[imp_cat][prod][cntr]['EU_import_new'] = (
                                                            y_prod)
                            y_prod -= y_prod

    def calc_reduc(self, dict_cf, dict_eb, df_tY_eu28):
        """ Calculate footprints for new EU28 imported final demand.

        Parameters
        ----------
        dict_cf: dictionary with characterisation factors of footprints
        dict_eb: dictionary with processed version of EXIOBASE
        df_tY_eu28: DataFrame with EU28 imported final demand

        """

        print('\nCalculating import embodied footprint reductions.')

        # Restructure new EU28 import data for DataFrames.
        dict_df_tY_import_new = {}
        for imp_cat_id, imp_cat in enumerate(self.dict_shift_result):
            dict_df_tY_import_new[imp_cat] = {}
            for prod in self.dict_shift_result[imp_cat]:
                for cntr in self.dict_shift_result[imp_cat][prod]:
                    dict_df_tY_import_new[imp_cat][(cntr, prod)] = (
                            self.dict_shift_result[imp_cat][prod][cntr]['EU_import_new'])

        # Calculate footprints for new EU28 imported final demand.
        dict_imp_new_reg = {}
        for imp_cat_id, imp_cat_sel in enumerate(dict_df_tY_import_new):
            df_tY_eu28_fdsum = df_tY_eu28.sum(axis=1)
            df_tY_eu28_import = df_tY_eu28_fdsum.copy()
            df_tY_eu28_import[:] = 0
            df_tY_eu28_import[list(
                    dict_df_tY_import_new[imp_cat_sel].keys())] = (
                    list(dict_df_tY_import_new[imp_cat_sel].values()))
            df_tY_eu28_import.columns = ['import']
            dict_df_imp_new = get_dict_df_imp(
                    dict_cf, dict_eb, df_tY_eu28_import)
            dict_imp_new = get_dict_imp(dict_df_imp_new)
            dict_imp_new_reg[imp_cat_sel] = {}
            for imp_cat_eff in dict_imp_new:
                if imp_cat_eff not in dict_imp_new_reg[imp_cat_sel]:
                    dict_imp_new_reg[imp_cat_sel][imp_cat_eff] = {}
                for prod in dict_imp_new[imp_cat_eff]:
                    if prod not in dict_imp_new_reg[imp_cat_sel][imp_cat_eff]:
                        dict_imp_new_reg[imp_cat_sel][imp_cat_eff][prod] = 0
                    for cntr in dict_imp_new[imp_cat_eff][prod]:
                        dict_imp_new_reg[imp_cat_sel][imp_cat_eff][prod] += (
                                dict_imp_new[imp_cat_eff][prod][cntr])

        # Calculate footprints for old EU28 imported final demand.
        dict_imp_cat_old = {}
        for imp_cat in self.dict_shift_result:
            dict_imp_cat_old[imp_cat] = {}
            for prod in self.dict_shift_result[imp_cat]:
                dict_imp_cat_old[imp_cat][prod] = 0
                for cntr in self.dict_shift_result[imp_cat][prod]:
                    imp_pME = self.dict_shift_result[imp_cat][prod][cntr]['imp_pME']
                    y_old = self.dict_shift_result[imp_cat][prod][cntr]['EU_import_old']
                    imp_abs = imp_pME*y_old
                    dict_imp_cat_old[imp_cat][prod] += imp_abs

        # Put footprints of new and old EU28 imported final demand in
        # dictionary.
        for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_new_reg):
            self.dict_reduc_result[imp_cat_sel] = {}
            for imp_cat_eff_id, imp_cat_eff in (
                    enumerate(dict_imp_new_reg[imp_cat_sel])):
                self.dict_reduc_result[imp_cat_sel][imp_cat_eff] = {}
                self.dict_reduc_result[imp_cat_sel][imp_cat_eff]['Ante'] = (
                        dict_imp_cat_old[imp_cat_eff])
                self.dict_reduc_result[imp_cat_sel][imp_cat_eff]['Post'] = (
                        dict_imp_new_reg[imp_cat_sel][imp_cat_eff])

    def calc(self, dict_cf, dict_eb, df_tY_eu28):
        """ Calculate reduction in import embodied footprints of EU28
            by source shifting

            Parameters
            ----------
            dict_cf: dictionary with characterisation factors of footprints
            dict_eb: dictionary with processed version of EXIOBASE
            df_tY_eu28: DataFrame with EU28 imported final demand

        """
        self.calc_shift(dict_cf, dict_eb, df_tY_eu28)
        self.calc_reduc(dict_cf, dict_eb, df_tY_eu28)


    def test(self):
        shift_dir_path = cfg.result_dir_path+cfg.shift_dir_name
        test_dir_path = shift_dir_path+cfg.test_dir_name
        test_export_file_name = 'test_export.txt'
        test_export_file_path = test_dir_path+test_export_file_name
        test_import_sum_file_name = 'test_import_sum.txt'
        test_import_sum_file_path = test_dir_path+test_import_sum_file_name

        error_export_bool = False
        error_import_sum_bool = False
        with open(test_export_file_path, 'w') as write_file:
            csv_export_file = csv.writer(write_file,
                                  delimiter='\t',
                                  lineterminator='\n')
            with open(test_import_sum_file_path, 'w') as write_file:
                csv_import_sum_file = csv.writer(write_file,
                      delimiter='\t',
                      lineterminator='\n')

                for imp_cat in self.dict_shift_result:
                    fp = self.dict_imp_cat_fp[imp_cat]
                    for prod in self.dict_shift_result[imp_cat]:
                        EU_import_old_sum = 0
                        EU_import_new_sum = 0
                        for cntr in self.dict_shift_result[imp_cat][prod]:
                            export = self.dict_shift_result[imp_cat][prod][cntr]['export']
                            EU_import_old = self.dict_shift_result[imp_cat][prod][cntr]['EU_import_old']
                            EU_import_new = self.dict_shift_result[imp_cat][prod][cntr]['EU_import_new']
                            EU_import_old_sum += EU_import_old
                            EU_import_new_sum += EU_import_new
                            if EU_import_new > export:
                                if not error_export_bool:
                                    csv_export_file.writerow(['New EU import larger than export sourcing country:'])
                                    csv_export_file.writerow(['Footprint',
                                                       'Product',
                                                       'Exporting country',
                                                       'Export',
                                                       'Old EU import',
                                                       'New EU import'])
                                error_export_bool = True
                                csv_export_file.writerow([fp,
                                                   prod,
                                                   cntr,
                                                   export,
                                                   EU_import_old,
                                                   EU_import_new])
                        if not(math.isclose(EU_import_old_sum, EU_import_new_sum)):
                            if not error_import_sum_bool:
                                csv_import_sum_file.writerow(['Sums of old and new EU import do not match:'])
                                csv_import_sum_file.writerow(['Footprint',
                                                              'Product',
                                                              'Exporting country',
                                                              'Export',
                                                              'Sum old EU import',
                                                              'Sum new EU import'])
                            error_import_sum_bool = True
                            csv_import_sum_file.writerow([fp,
                                                          prod,
                                                          cntr,
                                                          export,
                                                          EU_import_old_sum,
                                                          EU_import_new_sum])
                if not error_export_bool:
                    csv_export_file.writerow(['Success! New EU import never larger than export.'])

                if not error_import_sum_bool:
                    csv_import_sum_file.writerow(['Success! Sums of old and new EU import always match.'])


    def plot_shift(self):
        """ Plot source shifts of imports

        """
        dict_cntr_short_long = get_dict_cntr_short_long()

        shift_dir_path = cfg.result_dir_path+cfg.shift_dir_name
        pdf_dir_path = shift_dir_path+cfg.pdf_dir_name
        png_dir_path = shift_dir_path+cfg.png_dir_name

        print('\nSaving source shift plots to:\n\t{}\n\t{}'.format(
                png_dir_path,
                pdf_dir_path))

        # For both before and after source shifting, For each footprint,
        # for each product get list of labels from highest contributing
        # exporting country
        list_eu_import = ('EU_import_old', 'EU_import_new')
        dict_cntr_label = {}
        for eu_import in list_eu_import:
            dict_cntr_label[eu_import] = {}
            for imp_cat in self.dict_shift_result:
                dict_cntr_label[eu_import][imp_cat] = {}
                for prod in self.dict_shift_result[imp_cat]:
                    imp_abs_sum = 0
                    dict_cntr = {}
                    for cntr in self.dict_shift_result[imp_cat][prod]:
                        imp_pME = self.dict_shift_result[imp_cat][prod][cntr]['imp_pME']
                        y_new = self.dict_shift_result[imp_cat][prod][cntr][eu_import]
                        imp_abs = imp_pME*y_new
                        imp_abs_sum += imp_abs
                        dict_cntr[cntr] = imp_abs
                    list_imp_cat_prod_sort = sorted(
                            dict_cntr.items(),
                            key=operator.itemgetter(1), reverse=True)
                    list_imp_cat_prod_sort_trunc = []
                    if imp_abs_sum > 0:
                        imp_cum = 0
                        bool_add = True
                        for tup_cntr_imp in list_imp_cat_prod_sort:
                            cntr, imp_abs = tup_cntr_imp
                            imp_rel = imp_abs/imp_abs_sum
                            imp_cum += imp_rel
                            if imp_cum <= cfg.imp_cum_lim_source_shift:
                                list_imp_cat_prod_sort_trunc.append(cntr)

                            elif bool_add:
                                list_imp_cat_prod_sort_trunc.append(cntr)
                                bool_add = False
                    dict_cntr_label[eu_import][imp_cat][prod] = (
                            list_imp_cat_prod_sort_trunc)

        # Plot exports and current levels of EU28 imported final demand
        dict_lim = {}
        dict_ax = {}
        eu_import = 'EU_import_old'
        for imp_cat in self.dict_shift_result:
            dict_lim[imp_cat] = {}
            dict_ax[imp_cat] = {}
            for prod in self.dict_shift_result[imp_cat]:
                plt.close('all')
                x_start = 0
                y_start = 0
                list_rect_y = []
                list_rect_x = []
                fig = plt.figure(figsize=cm2inch((16, 8)), dpi=cfg.dpi)
                ax = plt.gca()
                for cntr in self.dict_shift_result[imp_cat][prod]:
                    imp_pME_prod_cntr = self.dict_shift_result[imp_cat][prod][cntr]['imp_pME']
                    y_prod_cntr = self.dict_shift_result[imp_cat][prod][cntr][eu_import]
                    x_prod_cntr = self.dict_shift_result[imp_cat][prod][cntr]['export']
                    if cntr in (dict_cntr_label[eu_import][imp_cat][prod]):

                        cntr_long = dict_cntr_short_long[cntr]
                        plt.text(x_start+y_prod_cntr/2,
                                 y_start+imp_pME_prod_cntr,
                                 ' '+cntr_long,
                                 rotation=90,
                                 verticalalignment='bottom',
                                 horizontalalignment='center',
                                 color='C0')
                    rect_y = patches.Rectangle((x_start, y_start),
                                               y_prod_cntr,
                                               imp_pME_prod_cntr)
                    rect_x = patches.Rectangle((x_start, y_start),
                                               x_prod_cntr,
                                               imp_pME_prod_cntr)
                    list_rect_y.append(rect_y)
                    list_rect_x.append(rect_x)
                    x_max = x_start+x_prod_cntr
                    y_max = y_start+imp_pME_prod_cntr
                    x_start += x_prod_cntr

                col_rect_y = mpl_col.PatchCollection(list_rect_y,
                                                     facecolor='C0')
                col_rect_x = mpl_col.PatchCollection(list_rect_x,
                                                     facecolor='gray')
                ax.add_collection(col_rect_x)
                ax.add_collection(col_rect_y)
                ax.autoscale()
                dict_lim[imp_cat][prod] = {}
                dict_lim[imp_cat][prod]['x'] = (0, x_max)
                dict_lim[imp_cat][prod]['y'] = (0, y_max)
                dict_ax[imp_cat][prod] = ax

        # Plot new EU28 imported final demand
        eu_import = 'EU_import_new'
        for imp_cat in self.dict_shift_result:
            for prod in self.dict_shift_result[imp_cat]:
                plt.close('all')
                x_start = 0
                y_start = 0
                list_rect_y = []
                ax = dict_ax[imp_cat][prod]
                for cntr in self.dict_shift_result[imp_cat][prod]:
                    imp_pME_prod_cntr = self.dict_shift_result[imp_cat][prod][cntr]['imp_pME']
                    y_prod_cntr_new = self.dict_shift_result[imp_cat][prod][cntr][eu_import]
                    x_prod_cntr = self.dict_shift_result[imp_cat][prod][cntr]['export']
                    rect_y = patches.Rectangle((x_start, y_start),
                                               y_prod_cntr_new,
                                               imp_pME_prod_cntr)
                    if cntr in dict_cntr_label[eu_import][imp_cat][prod]:
                        cntr_long = dict_cntr_short_long[cntr]

                        x_text = x_start+x_prod_cntr/2
                        y_text = y_start+imp_pME_prod_cntr
                        ax.text(x_text,
                                y_text,
                                ' '+cntr_long,
                                rotation=90,
                                verticalalignment='bottom',
                                horizontalalignment='center',
                                color='green')
                    list_rect_y.append(rect_y)
                    x_start += x_prod_cntr

                    rect_y = patches.Rectangle((0, 0), 0, 0)
                    list_rect_y.append(rect_y)

                col_rect_y = mpl_col.PatchCollection(list_rect_y,
                                                     facecolor='green')
                col_rect_y.set_alpha(0.5)
                ax.add_collection(col_rect_y)
                ax.autoscale()
                fig = ax.get_figure()
                unit = imp_cat[-1]
                ax.set_ylabel('{}/M€'.format(unit))
                ax.set_xlabel('M€')
                ax.set_xlim(dict_lim[imp_cat][prod]['x'])
                ax.set_ylim(dict_lim[imp_cat][prod]['y'])
                ax.locator_params(axis='both', nbins=4, tight=True)
                ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

                prod_short = self.dict_prod_long_short[prod]
                prod_short_lower = prod_short.lower()
                prod_short_lower_strip = prod_short_lower.strip()
                prod_short_lower_strip = prod_short_lower_strip.replace(':', '')
                prod_short_lower_strip_us = prod_short_lower_strip.replace(' ',
                                                                           '_')
                fp = self.dict_imp_cat_fp[imp_cat]
                fp_lower = fp.lower()
                fp_prod = fp_lower+'_'+prod_short_lower_strip_us
                fig.tight_layout(pad=0.1)

                pdf_file_name = fp_prod+'.pdf'
                pdf_file_path = pdf_dir_path+pdf_file_name
                fig.savefig(pdf_file_path)

                png_file_name = fp_prod+'.png'
                png_file_path = png_dir_path+png_file_name
                fig.savefig(png_file_path)

    def plot_reduc(self):
        """ Plot changes in footprints due to source shifting

        """

        reduc_dir_path = cfg.result_dir_path+cfg.reduc_dir_name
        pdf_dir_path = reduc_dir_path+cfg.pdf_dir_name
        png_dir_path = reduc_dir_path+cfg.png_dir_name

        dict_imp_cat_unit = get_dict_imp_cat_unit()
        list_prod_order_cons = get_list_prod_order_cons()

        print('\nSaving reduction plots to:\n\t{}\n\t{}'.format(
                png_dir_path,
                pdf_dir_path))


        # Calculate axis limits for plots of changed footprints of highest
        # contributing products for all footprints.
        dict_xlim_improv = {}
        for imp_cat_sel_id, imp_cat_sel in enumerate(self.dict_reduc_result):
            plt.close('all')
            fig = plt.figure(figsize=cm2inch((16, 1+13*0.4)), dpi=cfg.dpi)
            for imp_cat_eff_id, imp_cat_eff in (
                    enumerate(self.dict_reduc_result[imp_cat_sel])):
                plot_id = imp_cat_eff_id+1
                plot_loc = 140+plot_id
                ax = fig.add_subplot(plot_loc)
                df_old = pd.DataFrame(
                        self.dict_reduc_result[imp_cat_sel][imp_cat_eff]['Ante'],
                        index=['import'])
                df_new = pd.DataFrame(
                        self.dict_reduc_result[imp_cat_sel][imp_cat_eff]['Post'],
                        index=['import'])

                df = df_new-df_old
                df.rename(columns=self.dict_prod_long_short, inplace=True)
                df = df.reindex(list_prod_order_cons, axis=1)
                df_color = df.loc['import'] <= 0
                df.T.plot.barh(stacked=True,
                               ax=ax,
                               legend=False,
                               color=[df_color.map({True: 'g', False: 'r'})])
                if plot_id > 1:
                    ax.set_yticklabels([])
                plt.locator_params(axis='x', nbins=4)
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                xlim = ax.get_xlim()
                xlim_min_magn = 10**np.floor(np.log10(abs(xlim[0])))
                xlim_min_floor = (
                        math.floor(xlim[0]/xlim_min_magn)*xlim_min_magn)

                if xlim[1] > 0.0:
                    xlim_max_magn = 10**np.floor(np.log10(xlim[1]))
                    xlim_max_ceil = (
                            math.ceil(xlim[1]/xlim_max_magn)*xlim_max_magn)
                else:
                    xlim_max_ceil = int(xlim[1])
                tup_xlim_min_floor_max_ceil = (xlim_min_floor, xlim_max_ceil)

                if imp_cat_eff not in dict_xlim_improv:
                    dict_xlim_improv[imp_cat_eff] = tup_xlim_min_floor_max_ceil
                if xlim_min_floor < dict_xlim_improv[imp_cat_eff][0]:
                    xlim_new = tuple([xlim_min_floor,
                                      dict_xlim_improv[imp_cat_eff][1]])
                    dict_xlim_improv[imp_cat_eff] = xlim_new
                if xlim_max_ceil > dict_xlim_improv[imp_cat_eff][1]:
                    xlim_new = tuple([dict_xlim_improv[imp_cat_eff][0],
                                      xlim_max_ceil])
                    dict_xlim_improv[imp_cat_eff] = xlim_new

        # Plot changed footprints of highest contributing products for each
        # footprint.
        plt.close('all')
        for imp_cat_sel_id, imp_cat_sel in enumerate(self.dict_reduc_result):
            fig = plt.figure(
                    figsize=cm2inch((16, len(list_prod_order_cons)*.4+2)),
                    dpi=cfg.dpi)
            for imp_cat_eff_id, imp_cat_eff in (
                    enumerate(self.dict_reduc_result[imp_cat_sel])):
                plot_id = imp_cat_eff_id+1
                plot_loc = 140+plot_id
                ax = fig.add_subplot(plot_loc)
                fp = self.dict_imp_cat_fp[imp_cat_eff]
                ax.set_title(fp)
                unit = dict_imp_cat_unit[imp_cat_eff[-1]]
                ax.set_xlabel(unit, verticalalignment='baseline', labelpad=10)
                ax.set_xlim(dict_xlim_improv[imp_cat_eff])
                df_old = pd.DataFrame(
                        self.dict_reduc_result[imp_cat_sel][imp_cat_eff]['Ante'],
                        index=['import'])
                df_new = pd.DataFrame(
                        self.dict_reduc_result[imp_cat_sel][imp_cat_eff]['Post'],
                        index=['import'])
                df = df_new-df_old
                df.rename(columns=self.dict_prod_long_short, inplace=True)
                df = df.reindex(list_prod_order_cons, axis=1)
                df_color = df.loc['import'] <= 0
                df.T.plot.barh(stacked=True,
                               ax=ax,
                               legend=False,
                               color=[df_color.map({True: 'g', False: 'r'})],
                               width=0.8)

                yticklabels = ax.get_yticklabels()

                if plot_id > 1:
                    ax.set_yticklabels([])
                else:
                    ax.set_yticklabels(yticklabels)

                plt.locator_params(axis='x', nbins=4)
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                xtick_magnitude = self.dict_imp_cat_magnitude[imp_cat_eff]
                list_xtick = (
                        [i/xtick_magnitude for i in dict_xlim_improv[imp_cat_eff]])
                ax.set_xticks(list(dict_xlim_improv[imp_cat_eff]))
                ax.set_xticklabels(list_xtick)

                xtick_objects = ax.xaxis.get_major_ticks()
                xtick_objects[0].label1.set_horizontalalignment('left')
                xtick_objects[-1].label1.set_horizontalalignment('right')

            fig.tight_layout(pad=0)
            plt.subplots_adjust(wspace=0.1)
            fp = self.dict_imp_cat_fp[imp_cat_sel]
            fp_lower = fp.lower()

            fig_file_name = fp_lower+'.pdf'
            pdf_dir_path = reduc_dir_path+cfg.pdf_dir_name
            fig_file_path = pdf_dir_path+fig_file_name
            fig.savefig(fig_file_path)

            fig_file_name = fp_lower+'.png'
            png_dir_path = reduc_dir_path+cfg.png_dir_name
            fig_file_path = png_dir_path+fig_file_name
            fig.savefig(fig_file_path)


        # Calculate changed footprints aggregated over all products
        dict_pot_imp_agg = {}
        for imp_cat_sel_id, imp_cat_sel in enumerate(self.dict_reduc_result):
            fp_sel = self.dict_imp_cat_fp[imp_cat_sel]
            for imp_cat_eff_id, imp_cat_eff in enumerate(
                    self.dict_reduc_result[imp_cat_sel]):

                fp_eff = self.dict_imp_cat_fp[imp_cat_eff]
                if fp_eff not in dict_pot_imp_agg:
                    dict_pot_imp_agg[fp_eff] = {}
                if 'Ante' not in dict_pot_imp_agg[fp_eff]:
                    df_old = pd.DataFrame(
                            self.dict_reduc_result[imp_cat_sel][imp_cat_eff]['Ante'],
                            index=['import'])

                    df_old_sum = df_old.sum(axis=1)
                    dict_pot_imp_agg[fp_eff]['Prior'] = float(
                            df_old_sum['import'])
                df_new = pd.DataFrame(
                        self.dict_reduc_result[imp_cat_sel][imp_cat_eff]['Post'],
                        index=['import'])
                df_new_sum = df_new.sum(axis=1)
                dict_pot_imp_agg[fp_eff][fp_sel] = float(
                        df_new_sum['import'])

        # Calculate fraction of new and old footprints
        dict_pot_imp_agg_rel = {}
        for fp_eff_id, fp_eff in enumerate(dict_pot_imp_agg):
            list_imp_rel = []
            list_xticklabel = []
            for fp_sel in dict_pot_imp_agg[fp_eff]:
                imp_abs = dict_pot_imp_agg[fp_eff][fp_sel]
                if fp_sel == 'Prior':
                    imp_abs_prior = imp_abs
                else:
                    if fp_sel not in dict_pot_imp_agg_rel:
                        dict_pot_imp_agg_rel[fp_sel] = {}
                    imp_rel = imp_abs/imp_abs_prior
                    dict_pot_imp_agg_rel[fp_sel][fp_eff] = 1-imp_rel

        # Plot fraction of new and old footprint as spider plot.
        plt.close('all')
        fig = plt.figure(figsize=cm2inch((16, 16)), dpi=cfg.dpi)
        fp_order = ['Carbon', 'Land', 'Water', 'Material']
        for fp_sel_id, fp_sel in enumerate(dict_pot_imp_agg_rel):
            list_imp_rel = []
            list_xticklabel = []
            for fp_eff in fp_order:
                imp_rel = dict_pot_imp_agg_rel[fp_sel][fp_eff]
                list_imp_rel.append(imp_rel)
                list_xticklabel.append(fp_eff)

            plot_id = 220+fp_sel_id+1
            ax = fig.add_subplot(plot_id, projection='polar')
            ax.set_rticks([0.50, 1])
            ax.yaxis.set_ticklabels(['50%', '100%'])
            ax.set_rlabel_position(0)
            xtick_count = np.arange(2*math.pi/8,
                                    2*math.pi+2*math.pi/8,
                                    2*math.pi/4)
            ax.set_xticks(xtick_count)
            ax.set_xticklabels(list_xticklabel)
            ax.xaxis.set_tick_params(pad=10)
            ax.set_ylim([0, 1.0])
            y_val = list_imp_rel
            y_val.append(y_val[0])
            x_val = list(xtick_count)
            x_val.append(x_val[0])
            ax.plot(x_val, y_val, color='C2')
            ax_title = 'Optimized {} footprint'.format(fp_sel.lower())
            ax.set_title(ax_title)
        plt.tight_layout(pad=3)
        plot_name = 'spider_plot'
        pdf_file_name = plot_name+'.pdf'
        pdf_dir_path = reduc_dir_path+cfg.pdf_dir_name
        pdf_file_path = pdf_dir_path+pdf_file_name
        fig.savefig(pdf_file_path)
        png_file_name = plot_name+'.png'
        png_dir_path = reduc_dir_path+cfg.png_dir_name
        png_file_path = png_dir_path+png_file_name
        fig.savefig(png_file_path)

    def plot(self):
        """ Plot shifts in sourcing and changes in footprints due to these
            shifts.

        """
        self.plot_shift()
        self.plot_reduc()

    def log_shift(self):
        """ For each footprint, for each product, for each exporting country,
            write footprint intensity, export level, and old and new level of
            EU 28 imported final demand to file.

        """
        source_shift_dir_path = (cfg.result_dir_path
                                 + cfg.shift_dir_name)

        for imp_cat in self.dict_shift_result:
            fp = self.dict_imp_cat_fp[imp_cat]
            unit = imp_cat[-1]
            log_file_name = fp+'.txt'
            log_file_path = (source_shift_dir_path
                             + cfg.txt_dir_name
                             + log_file_name)
            print('\nSaving source shift log to:\n\t{}'.format(
                    source_shift_dir_path+cfg.txt_dir_name))

            with open(log_file_path, 'w') as write_file:
                csv_file = csv.writer(write_file,
                                      delimiter='\t',
                                      lineterminator='\n')
                csv_file.writerow(['Footprint',
                                   'Product',
                                   'Exporting country',
                                   unit+' per M Euro',
                                   'Export [M Euro]',
                                   'EU import ante [M Euro]',
                                   'EU import post [M Euro]'])

                for prod in self.dict_shift_result[imp_cat]:
                    for cntr in self.dict_shift_result[imp_cat][prod]:
                        imp_pME = self.dict_shift_result[imp_cat][prod][cntr]['imp_pME']
                        export = self.dict_shift_result[imp_cat][prod][cntr]['export']
                        import_old = self.dict_shift_result[imp_cat][prod][cntr]['EU_import_old']
                        import_new = self.dict_shift_result[imp_cat][prod][cntr]['EU_import_new']
                        csv_file.writerow([fp,
                                           prod,
                                           cntr,
                                           imp_pME,
                                           export,
                                           import_old,
                                           import_new])


    def log_reduc(self):
        """ For each optimized footprint, for each changed footprint, for each
            product, write footprint prior to and after source shifting.

        """
        reduc_dir_path = (cfg.result_dir_path
                          + cfg.reduc_dir_name)

        log_file_name = 'log.txt'
        log_file_path = (reduc_dir_path
                         + cfg.txt_dir_name
                         + log_file_name)

        print('\nSaving reduction log to:\n\t{}'.format(
                reduc_dir_path
                + cfg.txt_dir_name))

        with open(log_file_path, 'w') as write_file:
            csv_file = csv.writer(write_file,
                                  delimiter='\t',
                                  lineterminator='\n')
            csv_file.writerow(['Optimized footprint',
                               'Affected footprint',
                               'Product',
                               'Ante',
                               'Post'])
            for imp_cat_sel in self.dict_reduc_result:
                fp_sel = self.dict_imp_cat_fp[imp_cat_sel]
                for imp_cat_eff in self.dict_reduc_result[imp_cat_sel]:
                    fp_eff = self.dict_imp_cat_fp[imp_cat_eff]
                    for prod in self.dict_reduc_result[imp_cat_sel][imp_cat_eff]['Ante']:
                        fp_ante = self.dict_reduc_result[imp_cat_sel][imp_cat_eff]['Ante'][prod]
                        fp_post = self.dict_reduc_result[imp_cat_sel][imp_cat_eff]['Post'][prod]
                        csv_file.writerow([fp_sel,
                                           fp_eff,
                                           prod,
                                           fp_ante,
                                           fp_post])

    def log(self):
        """ Write old and new EU28 imported final demand and changed footprints
            to file.

        """
        self.log_shift()
        self.log_reduc()
