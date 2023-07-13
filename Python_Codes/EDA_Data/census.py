# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import os

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_dir = data_dir_base + "NASS_downloads/"

reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
# with open(NASS_dir+"2017_cdqt_data.txt") as f:
#     contents = f.readlines()

# f = open(NASS_dir+'2017_cdqt_data.txt','r')

# %%
# cdqt_data_2017 = pd.read_csv(NASS_dir + "2017_cdqt_data.txt", header=0, sep="	", on_bad_lines='skip')
# cdqt_data_2017

# cdqt_data_2017 = pd.read_csv(NASS_dir + "2017_cdqt_data.txt", header=0, sep=" ", on_bad_lines='skip')
# cdqt_data_2017

# %%

# %%
import pandas as pd
FarmOperation = pd.read_csv("/Users/hn/Documents/01_research_data/RangeLand/Data/NASS_downloads/FarmOperation.csv")


# %%

# %%
FarmOperation_BLACK_BELT = FarmOperation[FarmOperation["Ag District"]=="BLACK BELT"]
FarmOperation_BLACK_BELT

# %%

# %%
