import pickle
import os
import numpy

with open(os.path.join(os.path.dirname(__file__),'tests','results','ssp_on_orca.pkl'), "rb") as f:
    test_data = pickle.load(f)

print(test_data["5_humans"]["specifics"])