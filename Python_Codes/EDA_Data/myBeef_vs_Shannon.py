# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Old queries they said is not right. I had compared that to state level data from Shannon.
#
#  - Total Beef Cow inventory: https://quickstats.nass.usda.gov/#ADD6AB04-62EF-3DBF-9F83-5C23977C6DC7
#  - Inventory of Beef Cows:  https://quickstats.nass.usda.gov/#B8E64DC0-E63E-31EA-8058-3E86C0DCB74E
#  
#  --- 
# **New queries**
#
# Too many records. had to break it into two queries:
#
# Q1 and Q2 are very similar. I just did not choose the last part of first section Domain: (TOTAL vs INVENTORY of Beef COWs)
#
# In Q1 and Q2 rows are divided into different categories (1-9 head, 10-20 head). Too much work to clean it up., So, I choose Total in the domain and create Q4.
# __________________
#   - Q1_P1. https://quickstats.nass.usda.gov/#EDC639B8-9B16-3BC1-ABB7-8ACC6F9D6646
#   - Q1_P2. https://quickstats.nass.usda.gov/#BBF52292-3BBA-37B6-AE9C-379832BF2418
# __________________
#    - Q2_P1. https://quickstats.nass.usda.gov/#CEEE6107-8E75-3662-B8E0-5EE3E913F59E
#    - Q2_P2. https://quickstats.nass.usda.gov/#6F35D357-C7F7-340C-B2C2-34E285EE3147
# __________________
#    - Q3_P1. https://quickstats.nass.usda.gov/#0D301A69-C0D6-39AA-A5DF-E2EE46794D13
#    - Q3_P2. https://quickstats.nass.usda.gov/#24382805-7D70-3709-9AC9-6A0689FA721F
# __________________
#    - Q4. https://quickstats.nass.usda.gov/#E38831C2-2885-35D4-8A65-A152C5762BB5
# __________________
#    - Q5. https://quickstats.nass.usda.gov/#CBC0AA15-6F87-3A01-9C36-127DE111E760
#    

# %% [markdown]
# ## We should choose "Total":
#  - When Total is not chosen rows are broken down into different categories (1-10 head, 10-20 head).
#  - Other than item above, rows include Total as well. (so, we are concerned with Total anyway).
#  - Sum of all those rows w/ different categories, do not add up to "Total" probably because of "(D)" shit.

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"

dir_ = data_dir_base + "NASS_downloads/cow_inventory_Qs/"
reOrganized_dir = data_dir_base + "reOrganized/"
param_dir = data_dir_base + "parameters/"

# %%
Q4 = pd.read_csv(dir_ + "Q4.csv")
Q5 = pd.read_csv(dir_ + "Q5.csv")

# %%
# Q1_P1 = pd.read_csv(dir_ + "Q1_P1.csv", low_memory=False)
# Q1_P2 = pd.read_csv(dir_ + "Q1_P2.csv", low_memory=False)

# Q2_P1 = pd.read_csv(dir_ + "Q2_P1.csv", low_memory=False)
# Q2_P2 = pd.read_csv(dir_ + "Q2_P2.csv", low_memory=False)

# Q1 = pd.concat([Q1_P1, Q1_P2])
# Q2 = pd.concat([Q2_P1, Q2_P2])

# print (Q1.shape)
# Q1 = rc.clean_census(df=Q1, col_="Value")
# Q2 = rc.clean_census(df=Q2, col_="Value")
# print (Q1.shape)
# print ()
# print (f"{Q1.Domain.unique() = }")

# Q1_inv = Q1[(Q1.Domain == "INVENTORY OF BEEF COWS") & (Q1.County == "AUTAUGA") & (Q1.Year == 2017)].copy()
# Q1_total = Q1[(Q1.Domain == "TOTAL") & (Q1.County == "AUTAUGA") & (Q1.Year == 2017)].copy()

# Q1_inv.reset_index(drop=True, inplace=True)
# Q1_total.reset_index(drop=True, inplace=True)

# %%
Q4 = rc.clean_census(df = Q4, col_="Value")
Q5 = rc.clean_census(df = Q5, col_="Value")

# %%
SoI = ["Alabama", "Arkansas", "California", 
       "Colorado", "Florida", "Georgia", "Idaho",
       "Illinois", "Iowa", "Kansas", "Kentucky",
       "Louisiana", "Mississippi", "Missouri", "Montana", 
       "Nebraska", "New Mexico", "North Dakota", 
       "Oklahoma", "Oregon", "South Dakota", "Tennessee",
       "Texas", "Virginia", "Wyoming"]

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
state_25_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
Shannon_Beef_Cows_fromCATINV = pd.read_csv(reOrganized_dir + "Shannon_Beef_Cows_fromCATINV.csv")
Shannon_Beef_Cows_fromCATINV = Shannon_Beef_Cows_fromCATINV[Shannon_Beef_Cows_fromCATINV.state.isin(state_25_abb)]
Shannon_Beef_Cows_fromCATINV.reset_index(drop=True, inplace=True)
Shannon_Beef_Cows_fromCATINV.head(2)

# %%
shannon_years = [str(x) for x in np.arange(1997, 2018)]
cols = ["state"] + shannon_years
Shannon_Beef_Cows_fromCATINV = Shannon_Beef_Cows_fromCATINV[cols]
Shannon_Beef_Cows_fromCATINV.head(2)

# %%

# %%
