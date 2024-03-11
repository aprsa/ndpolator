#!/usr/bin/python

import re
import sys

with open(sys.argv[1], 'r') as f:
    print(re.sub(r'(?<!\$)\$(?!\$)', '@f$', f.read()))
