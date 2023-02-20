import pandas as pd
import os
basePath = os.path.dirname(__file__)
data = pd.read_json(open(basePath+'/pandas.json'))
print(data.to_string())