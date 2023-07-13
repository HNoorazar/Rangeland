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
meta_DF = FarmOperation[meta_cols].copy()
meta_DF.head(2)

# %% [markdown]
# # Alaska 
# has problem with ansi's

# %%
meta_DF[meta_DF['county_ansi'].isnull()]

values = {"county_ansi": 666}
meta_DF.fillna(value=values, inplace=True)

# %%
# meta_DF["county_ansi"] = meta_DF["county_ansi"].astype(int)
meta_DF[meta_DF['county_ansi'].isnull()]

# %%
meta_DF["county_ansi"] = meta_DF["county_ansi"].astype(int)

# %%
print (f"{meta_DF.shape = }")
print (f"{meta_DF.drop_duplicates().shape = }")

# %%
meta_DF.drop_duplicates(inplace=True)

# %%
meta_DF.to_csv(reOrganized_dir + "USDA_NASS_Census_metadata.csv")

# %%
AgLand.drop(bad_cols, axis="columns", inplace=True)
wetLand_area.drop(bad_cols, axis="columns", inplace=True)
feed_expense.drop(bad_cols, axis="columns", inplace=True)
FarmOperation.drop(bad_cols, axis="columns", inplace=True)

# %%
AgLand.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)
wetLand_area.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)
feed_expense.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)
FarmOperation.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)

# %%
AgLand

# %%
AgLand.to_csv(reOrganized_dir + "USDA_AgLand_cleaned_01.csv")
wetLand_area.to_csv(reOrganized_dir + "USDA_wetLand_area_cleaned_01.csv")
feed_expense.to_csv(reOrganized_dir + "USDA_feed_expense_cleaned_01.csv")
FarmOperation.to_csv(reOrganized_dir + "USDA_FarmOperation_cleaned_01.csv")

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
