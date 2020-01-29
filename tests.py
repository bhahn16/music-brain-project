# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:37:16 2019

@author: Ted
"""

import pytest
import helperFunctions as hf
text_file_example=r"sub-40_snl_l_enjoy_log.txt"
bad_text_file = "test.txt"
def test_opens_file():
    f = hf.load_txt_file(text_file_example)
    assert isinstance(f, _io.TextIOWrapper)
def test_bad_file():
    with pytest.raises(IOError):
        f = hf.load_txt_file(bad_text_file)
def test_splits():