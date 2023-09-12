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
# I want to have a direct stab at forming the datatables so they are usable for training the model.
# Let's just do it.

# %%
import pandas as pd
import numpy as np
import os

# %%
# import geopandas
# A = geopandas.read_file("/Users/hn/Desktop/amin/shapefile.shp")
# A.head(2)

# %% [markdown]
# According to Google Map the coordinate ```30.59375N 88.40625W``` is in Alabama. According to Bhupi's file it
# is in Alabama by state but in Mississipi by ```fips```. That place is on the border of the two states. I will go by fips.

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_dir = data_dir_base + "census/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_dir_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
file_name = "countyMean_seasonalVars_wFips.csv"
countyMean_seasonalVars = pd.read_csv(reOrganized_dir + "seasonal_variables/02_merged_mean_over_county/" + file_name)
print (f"{len(countyMean_seasonalVars.state.unique())=}")

# round numbers
countyMean_seasonalVars = countyMean_seasonalVars.round(decimals=2)
countyMean_seasonalVars.head(2)

# %%
countyMean_seasonalVars.head(2)

# %%
FIPS = pd.read_csv("/Users/hn/Documents/01_research_data/Sid/Analog/parameters/Min_counties.csv")
FIPS = FIPS[["state", "county", "fips"]]
FIPS.drop_duplicates(inplace=True)
FIPS.reset_index(drop=True, inplace=True)
FIPS.head(2)

# %%
SoI = ["Alabama", "Arkansas", "California", "Colorado", "Florida", "Georgia",
       "Idaho", "Illinois", "Iowa", "Kansas", "Kentucky", "Louisiana", "Mississippi",
       "Missouri", "Montana", "Nebraska", "New_Mexico",
       "North_Dakota", "Oklahoma", "Oregon", "South_Dakota", "Tennessee", "Texas", "Virginia", "Wyoming"]

# %%
Bhupi = pd.read_csv(param_dir + "Bhupi_25states_clean.csv")
Bhupi["SC"] = Bhupi.state + "-" + Bhupi.county
Bhupi.head(2)

# %% [markdown]
# ### Reshape seasonal variables.

# %%
# # 4 seasons will collapse into 1 row.
# L = int(len(countyMean_seasonalVars)/4)
# cntyMean_seasonVars_wide = pd.DataFrame(columns=["state", "county", "year"], index=range(L))
# cntyMean_seasonVars_wide.state = "A"
# cntyMean_seasonVars_wide.county = "A"
# cntyMean_seasonVars_wide.year = 0

# print (f"{cntyMean_seasonVars_wide.shape = }")
# cntyMean_seasonVars_wide.head(2)
# # countyMean_seasonalVars[["state", "county", "year"]].copy()

# season_list = ["S1", "S2", "S3", "S4"]
# temp_list = ["countyMean_avg_Tavg"] * 4
# temp_cols = [i + "_" + j for i, j in zip(season_list, temp_list)]

# precip_list = ["countyMean_total_precip"] * 4
# precip_cols = [i + "_" + j for i, j in zip(season_list, precip_list)]

# cntyMean_seasonVars_wide[precip_cols + temp_cols] = -60
# cntyMean_seasonVars_wide.head(2)

# countyMean_seasonalVars["state_county_year"] = countyMean_seasonalVars.state + "_" + \
#                                                countyMean_seasonalVars.county + "_" + \
#                                                countyMean_seasonalVars.year.astype("str")
# countyMean_seasonalVars.head(2)


# countyMean_seasonalVars[(countyMean_seasonalVars.state == "Alabama") & \
#                         (countyMean_seasonalVars.county == "Madison") & \
#                         (countyMean_seasonalVars.year == 1979)]

# %%
# # %%time
# wide_row_idx = 0
# for a_slice_patt in countyMean_seasonalVars.state_county_year.unique():
#     a_slice = countyMean_seasonalVars[countyMean_seasonalVars.state_county_year == a_slice_patt]

#     # populate the wide DF
#     cntyMean_seasonVars_wide.loc[wide_row_idx, "year"] = a_slice.year.iloc[0]
#     cntyMean_seasonVars_wide.loc[wide_row_idx, "state"] = a_slice.state.iloc[0]
#     cntyMean_seasonVars_wide.loc[wide_row_idx, "county"] = a_slice.county.iloc[0]
    
#     ### This is slow!!!
#     # cntyMean_seasonVars_wide.loc[wide_row_idx, temp_cols] = a_slice["countyMean_avg_Tavg"].values
#     # cntyMean_seasonVars_wide.loc[wide_row_idx, precip_cols] = a_slice["countyMean_total_precip"].values
    
#     cntyMean_seasonVars_wide.loc[wide_row_idx, temp_cols[0]] = a_slice["countyMean_avg_Tavg"].values[0]
#     cntyMean_seasonVars_wide.loc[wide_row_idx, temp_cols[1]] = a_slice["countyMean_avg_Tavg"].values[1]
#     cntyMean_seasonVars_wide.loc[wide_row_idx, temp_cols[2]] = a_slice["countyMean_avg_Tavg"].values[2]
#     cntyMean_seasonVars_wide.loc[wide_row_idx, temp_cols[3]] = a_slice["countyMean_avg_Tavg"].values[3]
    
#     cntyMean_seasonVars_wide.loc[wide_row_idx, precip_cols[0]] = a_slice["countyMean_total_precip"].values[0]
#     cntyMean_seasonVars_wide.loc[wide_row_idx, precip_cols[1]] = a_slice["countyMean_total_precip"].values[1]
#     cntyMean_seasonVars_wide.loc[wide_row_idx, precip_cols[2]] = a_slice["countyMean_total_precip"].values[2]
#     cntyMean_seasonVars_wide.loc[wide_row_idx, precip_cols[3]] = a_slice["countyMean_total_precip"].values[3]

#     wide_row_idx += 1

# cntyMean_seasonVars_wide = pd.merge(cntyMean_seasonVars_wide, 
#                                     Bhupi[["state", "county", "county_fips"]].drop_duplicates(), 
#                                     on=['state', 'county'], how='left')

# out_name = reOrganized_dir + "wide_seasonal_vars_cntyMean.csv"
# cntyMean_seasonVars_wide.to_csv(out_name, index = False)
# It took [25 minutes] to run this cell

# %%

# %%
cntyMean_seasonVars_wide = pd.read_csv(reOrganized_dir + "wide_seasonal_vars_cntyMean.csv")
cntyMean_seasonVars_wide.head(2)

# %%
feed_expense = pd.read_csv(reOrganized_dir + "USDA_feed_expense_cleaned_01.csv")
feed_expense.head(3)

# %%
len(feed_expense.state.unique())

# %%
cntyMean_seasonVars_wide_stateCounty = cntyMean_seasonVars_wide.state + "_" + cntyMean_seasonVars_wide.county
cntyMean_seasonVars_wide_stateCounty = list(cntyMean_seasonVars_wide_stateCounty.unique())
len(cntyMean_seasonVars_wide_stateCounty)

# %%
feed_expense_stateCounty = feed_expense.state + "_" + feed_expense.county
feed_expense_stateCounty = list(feed_expense_stateCounty.unique())
len(feed_expense_stateCounty)

# %%
missing = [x for x in cntyMean_seasonVars_wide_stateCounty if x not in feed_expense_stateCounty]

# %%
missing

# %%
sorted(list(feed_expense[feed_expense.state == "Alabama"].county.unique()))

# %%
sorted(list(cntyMean_seasonVars_wide[cntyMean_seasonVars_wide.state == "Alabama"].county.unique()))

# %%
sorted(list(feed_expense[feed_expense.state == "Arkansas"].county.unique()))

# %%
sorted(list(cntyMean_seasonVars_wide[cntyMean_seasonVars_wide.state == "Arkansas"].county.unique()))

# %%

# %%
Supriya = pd.read_csv(param_dir + "bad_grids_25states.csv")
Supriya.head(2)

# %%

# %%
sorted(list(Bhupi.state.unique()))

# %%

# %%
Bhupi_AR = Bhupi[Bhupi.state == "Arkansas"].copy()
sorted(list(Bhupi_AR.county.unique()))

# %%
Bhupi_TN = Bhupi[Bhupi.state == "Tennessee"].copy()
sorted(list(Bhupi_TN.county.unique()))

# %%
Bhupi_stateCounty = Bhupi.state + "_" + Bhupi.county
Bhupi_stateCounty = list(Bhupi_stateCounty.unique())
len(Bhupi_stateCounty)

# %%
missing = [x for x in Bhupi_stateCounty if x not in feed_expense_stateCounty]
missing

# %%
Bhupi_FL = Bhupi[Bhupi.state == "New Mexico"].copy()
feed_expense_FL = feed_expense[feed_expense.state == "New Mexico"].copy()

# %%
sorted(list(Bhupi_FL.county.unique()))

# %%
sorted(list(feed_expense_FL.county.unique()))

# %%
Bhupi.head(2)

# %%
