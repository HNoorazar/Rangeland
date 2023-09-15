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

# %% [markdown]
# - Feed expense by county, 1997-2017:
#
#    https://quickstats.nass.usda.gov/#EF899E9D-F162-3655-89D9-5C423132E97F
# __________________________________________________________________
#  
# - Acres enrolled in Conservation Reserve, Wetlands Reserve, Farmable Wetlands, or Conservation Reserve Enhancement Programs, 1997-2017:
#
#    https://quickstats.nass.usda.gov/#3A734A89-9829-3674-AFC6-C4764DF7B728
# __________________________________________________________________
#
# - Number of farm operations by county, 1997-2017:
#
#    https://quickstats.nass.usda.gov/#7310AC8E-D9CF-3BD9-8DC7-A4EF053FC56E
# __________________________________________________________________
#
# - Irrigated acres and total land in farms by county, 1997-2017:
#
#    https://quickstats.nass.usda.gov/#B2688D70-61AC-3E14-AA15-11882355E95E

# %%
import pandas as pd
import numpy as np
import os

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
USDA_data_dir = data_dir_base + "/NASS_downloads/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
USDA_files = [x for x in os.listdir(USDA_data_dir) if x.endswith(".csv")]
USDA_files

# %%
AgLand = pd.read_csv(USDA_data_dir + "AgLand.csv")
wetLand_area = pd.read_csv(USDA_data_dir + "wetLand_area.csv")
FarmOperation = pd.read_csv(USDA_data_dir + "FarmOperation.csv")
feed_expense = pd.read_csv(USDA_data_dir + "feed_expense.csv")

# %%
feed_expense.head(2)

# %%
feed_expense.Year.unique()

# %%
AgLand.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
wetLand_area.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
FarmOperation.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
feed_expense.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

sorted(list(feed_expense.columns))

# %%
print (f"{AgLand.shape = }")
print (f"{wetLand_area.shape = }")
print (f"{FarmOperation.shape = }")
print (f"{feed_expense.shape = }")

# %%
print ((feed_expense.columns == AgLand.columns).all())
print ((feed_expense.columns == wetLand_area.columns).all())
print ((feed_expense.columns == FarmOperation.columns).all())

# %%
print (AgLand.zip_code.unique())
print (wetLand_area.zip_code.unique())
print (feed_expense.zip_code.unique())
print (FarmOperation.zip_code.unique())
print ()
print (AgLand.week_ending.unique())
print (wetLand_area.week_ending.unique())
print (feed_expense.week_ending.unique())
print (FarmOperation.week_ending.unique())

# %%
print (AgLand.watershed.unique())
print (wetLand_area.watershed.unique())
print (feed_expense.watershed.unique())
print (FarmOperation.watershed.unique())
print ()
print (AgLand.domain_category.unique())
print (wetLand_area.domain_category.unique())
print (feed_expense.domain_category.unique())
print (FarmOperation.domain_category.unique())

# %%
FarmOperation.state_ansi.unique()

# %%
print (AgLand.domain.unique())
print (wetLand_area.domain.unique())
print (feed_expense.domain.unique())
print (FarmOperation.domain.unique())
print ()

print (AgLand.watershed_code.unique())
print (wetLand_area.watershed_code.unique())
print (feed_expense.watershed_code.unique())
print (FarmOperation.watershed_code.unique())

# %%
print (AgLand.ag_district_code.unique())
print (wetLand_area.ag_district_code.unique())
print (feed_expense.ag_district_code.unique())
print (FarmOperation.ag_district_code.unique())

# %%
print (AgLand.watershed.unique())
print (wetLand_area.watershed.unique())
print (feed_expense.watershed.unique())
print (FarmOperation.watershed.unique())

# %%
print (AgLand.region.unique())
print (wetLand_area.region.unique())
print (feed_expense.region.unique())
print (FarmOperation.region.unique())

# %%
print (AgLand.program.unique())
print (wetLand_area.program.unique())
print (feed_expense.program.unique())
print (FarmOperation.program.unique())

# %%
print (AgLand.period.unique())
print (wetLand_area.period.unique())
print (feed_expense.period.unique())
print (FarmOperation.period.unique())
print ()
print (AgLand.geo_level.unique())
print (wetLand_area.geo_level.unique())
print (feed_expense.geo_level.unique())
print (FarmOperation.geo_level.unique())

# %%
print (AgLand.period.unique())
print (wetLand_area.period.unique())
print (feed_expense.period.unique())
print (FarmOperation.period.unique())
print ()
print (AgLand.data_item.unique())
print (wetLand_area.data_item.unique())
print (feed_expense.data_item.unique())
print (FarmOperation.data_item.unique())

# %%

# %%
bad_cols  = ["watershed", "watershed_code", 
             "domain", "domain_category", 
             "region", "period",
             "week_ending", "zip_code", "program", "geo_level"]


meta_cols = ["state", "county", "county_ansi", "state_ansi", "ag_district_code"]

# %%
FarmOperation['county_ansi'].fillna(666, inplace=True)

FarmOperation["state_ansi"] = FarmOperation["state_ansi"].astype('int32')
FarmOperation["county_ansi"] = FarmOperation["county_ansi"].astype('int32')

FarmOperation["state_ansi"] = FarmOperation["state_ansi"].astype('str')
FarmOperation["county_ansi"] = FarmOperation["county_ansi"].astype('str')

FarmOperation.state = FarmOperation.state.str.title()
FarmOperation.county = FarmOperation.county.str.title()

# %%
FarmOperation[["state", "county", "state_ansi", "county_ansi"]].head(5)

# %%
for idx in FarmOperation.index:
    if len(FarmOperation.loc[idx, "state_ansi"]) == 1:
        FarmOperation.loc[idx, "state_ansi"] = "0" + FarmOperation.loc[idx, "state_ansi"]
        
    if len(FarmOperation.loc[idx, "county_ansi"]) == 1:
        FarmOperation.loc[idx, "county_ansi"] = "00" + FarmOperation.loc[idx, "county_ansi"]
    elif len(FarmOperation.loc[idx, "county_ansi"]) == 2:
        FarmOperation.loc[idx, "county_ansi"] = "0" + FarmOperation.loc[idx, "county_ansi"]

# %%
FarmOperation[["state", "county", "state_ansi", "county_ansi"]].head(5)

# %%
FarmOperation["county_fips"] = FarmOperation["state_ansi"] + FarmOperation["county_ansi"] 

# %%
meta_DF = FarmOperation[meta_cols].copy()
meta_DF.head(2)

# %% [markdown]
# # Alaska 
# has problem with ansi's

# %%
# meta_DF[meta_DF['county_ansi'].isnull()]

# values = {"county_ansi": 666}
# meta_DF.fillna(value=values, inplace=True)

# %%
# meta_DF["county_ansi"] = meta_DF["county_ansi"].astype(int)
meta_DF[meta_DF['county_ansi'].isnull()]

# %%
# meta_DF["county_ansi"] = meta_DF["county_ansi"].astype(int)

# %%
print (f"{meta_DF.shape = }")
print (f"{meta_DF.drop_duplicates().shape = }")

# %%
meta_DF.drop_duplicates(inplace=True)
meta_DF.head(2)

# %%
meta_DF.to_csv(reOrganized_dir + "USDA_NASS_Census_metadata.csv", index=False)

# %%

# %%
AgLand.drop(bad_cols, axis="columns", inplace=True)
wetLand_area.drop(bad_cols, axis="columns", inplace=True)
feed_expense.drop(bad_cols, axis="columns", inplace=True)
FarmOperation.drop(bad_cols, axis="columns", inplace=True)

# %%

# %%
AgLand.head(2)

# %%
# AgLand[AgLand['county_ansi'].isnull()]

# %%
feed_expense[(feed_expense.state=="CALIFORNIA") & (feed_expense.county=="SAN FRANCISCO")]

# %%

# %%

# %%
AgLand['county_ansi'].fillna(666, inplace=True)

AgLand["state_ansi"] = AgLand["state_ansi"].astype('int32')
AgLand["county_ansi"] = AgLand["county_ansi"].astype('int32')

AgLand["state_ansi"] = AgLand["state_ansi"].astype('str')
AgLand["county_ansi"] = AgLand["county_ansi"].astype('str')

AgLand.state = AgLand.state.str.title()
AgLand.county = AgLand.county.str.title()

for idx in AgLand.index:
    if len(AgLand.loc[idx, "state_ansi"]) == 1:
        AgLand.loc[idx, "state_ansi"] = "0" + AgLand.loc[idx, "state_ansi"]
        
    if len(AgLand.loc[idx, "county_ansi"]) == 1:
        AgLand.loc[idx, "county_ansi"] = "00" + AgLand.loc[idx, "county_ansi"]
    elif len(AgLand.loc[idx, "county_ansi"]) == 2:
        AgLand.loc[idx, "county_ansi"] = "0" + AgLand.loc[idx, "county_ansi"]

AgLand["county_fips"] = AgLand["state_ansi"] + AgLand["county_ansi"]
AgLand[["state", "county", "state_ansi", "county_ansi", "county_fips"]].head(5)

# %%

# %%
wetLand_area['county_ansi'].fillna(666, inplace=True)

wetLand_area["state_ansi"] = wetLand_area["state_ansi"].astype('int32')
wetLand_area["county_ansi"] = wetLand_area["county_ansi"].astype('int32')

wetLand_area["state_ansi"] = wetLand_area["state_ansi"].astype('str')
wetLand_area["county_ansi"] = wetLand_area["county_ansi"].astype('str')

wetLand_area.state = wetLand_area.state.str.title()
wetLand_area.county = wetLand_area.county.str.title()

for idx in wetLand_area.index:
    if len(wetLand_area.loc[idx, "state_ansi"]) == 1:
        wetLand_area.loc[idx, "state_ansi"] = "0" + wetLand_area.loc[idx, "state_ansi"]
        
    if len(wetLand_area.loc[idx, "county_ansi"]) == 1:
        wetLand_area.loc[idx, "county_ansi"] = "00" + wetLand_area.loc[idx, "county_ansi"]
    elif len(wetLand_area.loc[idx, "county_ansi"]) == 2:
        wetLand_area.loc[idx, "county_ansi"] = "0" + wetLand_area.loc[idx, "county_ansi"]

wetLand_area["county_fips"] = wetLand_area["state_ansi"] + wetLand_area["county_ansi"]
wetLand_area[["state", "county", "state_ansi", "county_ansi", "county_fips"]].head(5)

# %%
feed_expense['county_ansi'].fillna(666, inplace=True)

feed_expense["state_ansi"] = feed_expense["state_ansi"].astype('int32')
feed_expense["county_ansi"] = feed_expense["county_ansi"].astype('int32')

feed_expense["state_ansi"] = feed_expense["state_ansi"].astype('str')
feed_expense["county_ansi"] = feed_expense["county_ansi"].astype('str')

feed_expense.state = feed_expense.state.str.title()
feed_expense.county = feed_expense.county.str.title()

for idx in feed_expense.index:
    if len(feed_expense.loc[idx, "state_ansi"]) == 1:
        feed_expense.loc[idx, "state_ansi"] = "0" + feed_expense.loc[idx, "state_ansi"]
        
    if len(feed_expense.loc[idx, "county_ansi"]) == 1:
        feed_expense.loc[idx, "county_ansi"] = "00" + feed_expense.loc[idx, "county_ansi"]
    elif len(feed_expense.loc[idx, "county_ansi"]) == 2:
        feed_expense.loc[idx, "county_ansi"] = "0" + feed_expense.loc[idx, "county_ansi"]
        
        
feed_expense["county_fips"] = feed_expense["state_ansi"] + feed_expense["county_ansi"]
feed_expense[["state", "county", "state_ansi", "county_ansi", "county_fips"]].head(5)

# %%
feed_expense[(feed_expense.state=="Alabama") & (feed_expense.county=="Washington")]

# %%
# AgLand.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)
# wetLand_area.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)
# feed_expense.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)
# FarmOperation.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)

# %%
# AgLand.to_csv(reOrganized_dir + "USDA_AgLand_cleaned_01.csv", index=False)
# wetLand_area.to_csv(reOrganized_dir  + "USDA_wetLand_area_cleaned_01.csv",  index=False)
# feed_expense.to_csv(reOrganized_dir  + "USDA_feed_expense_cleaned_01.csv",  index=False)
# FarmOperation.to_csv(reOrganized_dir + "USDA_FarmOperation_cleaned_01.csv", index=False)


import pickle
from datetime import datetime

filename = reOrganized_dir + "USDA_data.sav"

export_ = {"AgLand": AgLand, "wetLand_area": wetLand_area, 
           "feed_expense": feed_expense, "FarmOperation": FarmOperation, 
           "source_code" : "00_clean_USDA_data_addFIPS",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
feed_expense.county_fips[1]

# %% [markdown]
# # Tonsor
# - CRP
# - Feed Expense
# - Population/county (**missing. need to contact Census Bureau**)
# - Percentage of irrigated acres
# - FarmOperation not needed. NASS guy had created this.

# %%
CRP = wetLand_area.copy()

# %%
AgLand.head(5)

# %%
CRP.head(5)

# %%
feed_expense.head(5)

# %%

# %%

# %%

# %%
