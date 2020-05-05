"""
Tompkins County cancer data from:

The New York State Cancer Surveillance Improvement Initiative (CSII)
http://www.health.ny.gov/diseases/cancer/csii/
http://www.health.ny.gov/statistics/cancer/registry/zipcode/index.htm

The data cover 2005 - 2009.

For town names ending with "*", note:

    * This ZIP Code crosses county boundaries. The values provided are for
    the entire ZIP Code, not just the portion in this county.

The web site notes: "Incidence data are provisional, November 2011."

2020-05-04 Revised for BDA20 by Tom Loredo
"""

import io

import numpy as np
import scipy
from numpy import *
from scipy import stats


# The next string shows the raw text; we use delimted text for parsing, below.

# Columns:
# Primary ZIP Code
# Post Office
# Included ZIP Codes
# Number of Cases Observed
# Number of Cases Expected
# Percent Difference from Expected

breast_data_txt = \
    """13053   Dryden* 13784   15  15.5    Within 15% of expected
13068   Freeville       21  18.5    Within 15% of expected
13073   Groton  13102   24  23.5    Within 15% of expected
13736   Berkshire*      10  9.1 Within 15% of expected
13864   Willseyville*       4   4.0 Very sparse data
14817   Brooktondale*   14881   13  8.5 More than 50% above expected
14850   Ithaca  13062, 14851, 14852, 14853  181 175.1   Within 15% of expected
14867   Newfield        26  18.7    15 to 49% above expected
14882   Lansing*        11  14.6    15 to 50% below expected
14883   Spencer*        14  13.6    Within 15% of expected
14886   Trumansburg*    14854, 14863    30  26.4    Within 15% of expected
"""

# Converted to CSV with "|"" delim using:
# http://www.convertcsv.com/html-table-to-csv.htm

# Breast cancer data:
bc_data_csv = io.StringIO(
"""Primary ZIP Code|Post Office|Included ZIP Codes|Number of Cases Observed|Number of Cases Expected|Percent Difference from Expected
13053|Dryden*|13784|15|15.5|Within 15% of expected
13068|Freeville||21|18.5|Within 15% of expected
13073|Groton|13102|24|23.5|Within 15% of expected
13736|Berkshire*||10|9.1|Within 15% of expected
13864|Willseyville*||4|4.0|Very sparse data
14817|Brooktondale*|14881|13|8.5|More than 50% above expected
14850|Ithaca|13062, 14851, 14852, 14853|181|175.1|Within 15% of expected
14867|Newfield||26|18.7|15 to 49% above expected
14882|Lansing*||11|14.6|15 to 50% below expected
14883|Spencer*||14|13.6|Within 15% of expected
14886|Trumansburg*|14854, 14863|30|26.4|Within 15% of expected
""")

# Prostate cancer data:
pc_data_csv = io.StringIO(
"""Primary ZIP Code|Post Office|Included ZIP Codes|Number of Cases Observed|Number of Cases Expected|Percent Difference from Expected
13053|Dryden*|13784|27|18.5|15 to 49% above expected
13068|Freeville||25|22.1|Within 15% of expected
13073|Groton|13102|31|27.0|Within 15% of expected
13736|Berkshire*||12|11.6|Within 15% of expected
13864|Willseyville*||3|4.9|Very sparse data
14817|Brooktondale*|14881|12|10.1|15 to 49% above expected
14850|Ithaca|13062, 14851, 14852, 14853|240|188.7|15 to 49% above expected
14867|Newfield||25|21.9|Within 15% of expected
14882|Lansing*||23|18.1|15 to 49% above expected
14883|Spencer*||10|16.5|15 to 50% below expected
14886|Trumansburg*|14854, 14863|54|30.0|More than 50% above expected
""")

sep = '|'

ith_indx = 6

bc_data = loadtxt(bc_data_csv, delimiter=sep, skiprows=1, usecols=[3,4])
bc_counts = asarray(bc_data[:,0], dtype=int)
bc_expect = bc_data[:,1]

pc_data = loadtxt(pc_data_csv, delimiter=sep, skiprows=1, usecols=[3,4])
pc_counts = asarray(pc_data[:,0], dtype=int)
pc_expect = pc_data[:,1]
