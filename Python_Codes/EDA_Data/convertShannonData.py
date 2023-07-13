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

reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %% [markdown]
# ## We just need sheet A (beef cows) from CATINV

# %%
# CATINV = pd.read_excel(io=param_dir + "CATINV.xlsx", sheet_name=0)
xl = pd.ExcelFile(Shannon_data_dir + "CATINV.xls")
EX_sheet_names = xl.sheet_names  # see all sheet names
EX_sheet_names = EX_sheet_names[1:]
EX_sheet_names

# %%
CATINV_df = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

# %%
ii = 0
sheet_name_ = EX_sheet_names[ii]

curr_sheet = pd.read_excel(io = Shannon_data_dir + "CATINV.xls", 
                           sheet_name = sheet_name_, 
                           header = 0, skiprows = 0)
curr_sheet_columns = list(curr_sheet.columns)
named_columns = curr_sheet_columns[0] # [x for x in curr_sheet_columns if not("Unnamed" in x)]
named_columns

# %%
curr_sheet.columns = list(curr_sheet.iloc[1, ].astype(str))
curr_sheet = curr_sheet[2:].copy()
curr_sheet.rename({'nan': 'state'}, axis=1, inplace=True)
curr_sheet.rename(columns={x: x.replace('.0', '') for x in curr_sheet.columns[1:]}, inplace=True)
curr_sheet.reset_index(drop=True, inplace=True)
curr_sheet.loc[:, curr_sheet.columns[1]:curr_sheet.columns[-1]] = curr_sheet.loc[:, \
                                                curr_sheet.columns[1]:curr_sheet.columns[-1]]*1000

# Drop rows that are entirely NA
curr_sheet.dropna(axis=0, how = 'all', inplace = True)

# Drop rows where state is NA
curr_sheet.dropna(subset=['state'], inplace = True)
Beef_Cows_CATINV = curr_sheet.copy()
Beef_Cows_CATINV.tail(4)

# %%
out_name = reOrganized_dir + "Beef_Cows_fromCATINV.csv"
Beef_Cows_CATINV.to_csv(out_name, index = False)

# %% [markdown]
# ### We just need sheet A (beef cows) from ```Annual Cattle Inventory by State.xlsx```

# %%
xl = pd.ExcelFile(Shannon_data_dir + "Annual Cattle Inventory by State.xls")
EX_sheet_names = xl.sheet_names  # see all sheet names
EX_sheet_names = EX_sheet_names[1:]
EX_sheet_names

# %%
ii = 0
sheet_name_ = EX_sheet_names[ii]

curr_sheet = pd.read_excel(io = Shannon_data_dir + "Annual Cattle Inventory by State.xls", 
                           sheet_name = sheet_name_, 
                           header = 0, skiprows = 0)
curr_sheet_columns = list(curr_sheet.columns)
named_columns = curr_sheet_columns[0] # [x for x in curr_sheet_columns if not("Unnamed" in x)]

print (f"{named_columns=}")
curr_sheet.head(4)

# %%
curr_sheet.columns = list(curr_sheet.iloc[1, ].astype(str))
curr_sheet = curr_sheet[2:].copy()

curr_sheet.rename({'nan': 'state'}, axis=1, inplace=True)
curr_sheet.rename(columns={x: x.replace('.0', '') for x in curr_sheet.columns[1:]}, inplace=True)
curr_sheet.reset_index(drop=True, inplace=True)

curr_sheet.loc[:, curr_sheet.columns[1]:curr_sheet.columns[-1]] = curr_sheet.loc[:, \
                                                curr_sheet.columns[1]:curr_sheet.columns[-1]]*1000


# Drop rows that are entirely NA
curr_sheet.dropna(axis=0, how = 'all', inplace = True)

# Drop rows where state is NA
curr_sheet.dropna(subset=['state'], inplace = True)
Beef_Cows_annual = curr_sheet.copy()
Beef_Cows_annual.head(2)

# %%
Beef_Cows_annual.tail(4)

# %%
out_name = reOrganized_dir + "Beef_Cows_fromAnnualCattleInventorybyState.csv"
Beef_Cows_CATINV.to_csv(out_name, index = False)

# %%
print (f"{Beef_Cows_CATINV.shape=}")
print (f"{Beef_Cows_annual.shape=}")

# %%
Beef_Cows_CATINV.head(4)

# %%
Beef_Cows_annual.head(4)

# %%
Beef_Cows_annual.loc[:, "1920":"2020"].equals(Beef_Cows_CATINV.loc[:, "1920":"2020"])

# %%
Beef_Cows_annual.loc[:, "2021"] - (Beef_Cows_CATINV.loc[:, "2021"])

# %%

# %% [markdown]
# ## We just need sheet B (beef cows) from Weekly Regional Cow Slaughter

# %%
file_ = "Weekly Regional Cow Slaughter.xls"
xl = pd.ExcelFile(Shannon_data_dir + file_)
EX_sheet_names = xl.sheet_names  # see all sheet names
EX_sheet_names = EX_sheet_names[1:]
EX_sheet_names

# %%
ii = 0
sheet_name_ = EX_sheet_names[ii]

curr_sheet = pd.read_excel(io = Shannon_data_dir + file_, 
                           sheet_name = sheet_name_, 
                           header = 0, skiprows = 4)
curr_sheet_columns = list(curr_sheet.columns)
curr_sheet.head(7)

# %%
curr_sheet.loc[1, ] = curr_sheet.loc[0, ] + curr_sheet.loc[1, ]
curr_sheet = curr_sheet.loc[1:, ].copy()
curr_sheet.reset_index(drop=True, inplace=True)
curr_sheet.drop(axis=1, index=1, inplace=True)
curr_sheet.reset_index(drop=True, inplace=True)

for a_col in list(curr_sheet.columns):
    if not ("Unnamed" in a_col):
        curr_index = list(curr_sheet.columns).index(a_col)
        new_part = a_col.replace(".1", "").replace("- ", "").replace(" ", "_").replace("(", "").replace(")", "")
        
        curr_sheet.iloc[0, curr_index]= new_part + "_" + curr_sheet.iloc[0, curr_index].replace(" ", "")
        curr_sheet.iloc[0, curr_index+1] = new_part + "_" + curr_sheet.iloc[0, curr_index+1].replace(" ", "")

curr_sheet.iloc[0, 0] = "date"
curr_sheet.iloc[0, 1] = "week"
curr_sheet.rename(columns=curr_sheet.iloc[0], inplace = True)

# Drop first row
curr_sheet.drop(axis=1, index=0, inplace=True)
curr_sheet.reset_index(drop=True, inplace=True)

curr_sheet.head(7)

# %%
curr_sheet["Region_1_&_Region_2_beef"] = curr_sheet["Region_1_&_Region_2_Beef&dairy"] - \
                                                   curr_sheet["Region_1_&_Region_2_dairy"]

for ii in range(3,11):
    curr_sheet["Region_" + str(ii) + "_beef"] = curr_sheet["Region_" + str(ii) + "_Beef&dairy"] - \
                                                   curr_sheet["Region_" + str(ii) + "_dairy"]

# %%
out_name = reOrganized_dir + "Beef_Cows_fromWeeklyRegionalCowSlaughter.csv"
curr_sheet.to_csv(out_name, index = False)

# %%

# %%

# %%

# %%

# %%

# %%
