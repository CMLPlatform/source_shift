# -*- coding: utf-8 -*-
""" Main script for paper on
    reducing import embedded footprints of EU28 by source shifting
    Copyright (C) 2018

    Bertram F. de Boer
    Faculty of Science
    Institute of Environmental Sciences (CML)
    Department of Industrial Ecology
    Einsteinweg 2
    2333 CC Leiden

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
