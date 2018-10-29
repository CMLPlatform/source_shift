# -*- coding: utf-8 -*-
""" Main script for paper on
    reducing import embedded footprints of EU28 by source shifting
"""

import utils as ut

# Load EXIOBASE.
dict_eb = ut.get_dict_eb()

# Load characterisation factors.
dict_cf = ut.get_dict_cf(dict_eb)

# Load EU28 imported final demand.
df_tY_eu28 = ut.get_df_tY_eu28(dict_eb)

# Calculate highest contributing products for each footprint
priority = ut.Priority()
priority.calc(dict_cf, dict_eb, df_tY_eu28)
priority.plot()
priority.log()

# Calculate reduction in import embedded footprints of EU28 by source shifting
source_shift = ut.SourceShift()
source_shift.calc(dict_cf, dict_eb, df_tY_eu28)
source_shift.plot()
source_shift.log()
