# -*- coding: utf-8 -*-
""" Configurations for script of paper on
    reducing import embedded footprints of EU28 by source shifting

"""

import datetime
import matplotlib.pyplot as plt


def get_date():
    """ Get string with date.
        Used to make result directories.

        Returns:
        --------
        string with date.
    """

    date = datetime.datetime.now()
    return '{}{:02}{:02}'.format(date.year, date.month, date.day)


date = get_date()

"""Define directory names to store results."""
method = '_source_shift/'
result_dir_path = 'result/'+date+method
priority_setting_dir_name = '1_priority_setting/'
shift_dir_name = '2_shift/'
reduc_dir_name = '3_reduction/'
pdf_dir_name = 'pdf/'
png_dir_name = 'png/'
txt_dir_name = 'txt/'
list_output_dir_name = [pdf_dir_name, png_dir_name, txt_dir_name]

"""Define directory and file names from which to read data."""
# Root directory for data.
data_path = 'data/'

# Directory with raw text version of EXIOBASE. Used for parsing.
eb_path=data_path + 'mrIOT_pxp_ita_transactions_3.3_2011/'

# Name of processed EXIOBASE pickle.
dict_eb_file_name = 'dict_eb_proc.pkl'

# Boolean to save processed EXIOBASE version for future uses.
save_eb = True

# File with country codes of EU28 countries.
# Used to select final demand matrix.
eu28_file_name = 'EU28.txt'

# Files with names of footprints. Used to select characterization factors.
e_fp_file_name = 'list_impact_emission.txt'
m_fp_file_name = 'list_impact_material.txt'
r_fp_file_name = 'list_impact_resource.txt'

# Files with characterization factors of footprints.
cQe_file_name = 'Q_emission.txt'
cQm_file_name = 'Q_material.txt'
cQr_file_name = 'Q_resource.txt'

# File linking country codes with country names. Used for plotting.
country_code_file_name = 'country_codes.txt'

# File with long versions of product names. Used for plotting.
prod_long_file_name = 'prod_long.txt'

# File with short versions of product names. Used for plotting.
prod_short_file_name = 'prod_short.txt'

# File linking names of impact categories with names of footprints.
# Used for plotting.
cf_long_footprint_file_name = 'cf_long_footprint.txt'

# File linking impact categories with orders of magnitude.
# Used for plotting.
cf_magnitude_file_name = 'cf_magnitude.txt'

# File with concatinated list of highest contributing products for each
# footprint.
# Used for plotting.
prod_order_file_name = 'prod_order.txt'

# Set font size for plotting.
font_size = 8.0
plt.rc('mathtext', default='regular')
plt.rc('font', size=font_size)
plt.rc('axes', titlesize=font_size)

# Limit of cumulative relative footprint for highest contributing products.
imp_cum_lim_priority = 0.5

# Limit of cumulative relative footprint for all products.
# This is set to 110% to include all products.
# If this is set to 100%, not all products are included due to rounding errors.
imp_cum_lim_full = 1.1

# Limit of cumulative relative footprint in source shifting.
# Used to select country names for plotting.
imp_cum_lim_source_shift = 0.5

# Transparency of overlayed new EU28 imported final demand
# Used for plotting.
reduc_alpha = 0.5

# Threshold for export level. If country exports less than this amount for a
# product, country is not considered for source shifting due to noise.
x_prod_cntr_min = 0.5
