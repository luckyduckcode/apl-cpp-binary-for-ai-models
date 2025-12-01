#!/usr/bin/env python3
import runpy
import os
base = os.path.dirname(__file__)
runpy.run_path(os.path.join(base, 'quantized_test_full.py'), run_name='__main__')
