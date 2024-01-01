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
# ### Variables we need:
#  - ```Inventory```
#  - ```NPP```
#  - ```RA?```
#  - ```herb?```
#  - ```feed cost``` (in some form)
#  - ```resident population```
#  - ```irrigated/dryland area``` or just rangeland area (as percentage)
#  
# **Herath** 
#  - ```slaughter numbers```
#  - ```weather variables```
#  - ```unemployment rate```
#  - ```farmland availibility```
#  - ```labor cost```
#  - ```energy price```
#  - ```value of farmland```
#  
# **Kirti**
#  - ```Time of max NPP```
#  - ```variation of NPP```

# %%
import shutup

shutup.please()

import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
import statistics
import statsmodels.api as sm

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
# for bold print
start_b = "\033[1m"
end_b = "\033[0;0m"
print ("This is " + start_b + "a_bold_text" + end_b + "!")

# %% [markdown]
# # Read the data

# %%
SoI = [
    "Alabama",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Florida",
    "Georgia",
    "Idaho",
    "Illinois",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Mexico",
    "North Dakota",
    "Oklahoma",
    "Oregon",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Virginia",
    "Washington",
    "Wyoming",
]

abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI_abb = []
for x in SoI:
    SoI_abb = SoI_abb + [abb_dict["full_2_abb"][x]]

# %% [markdown]
# #### County FIPS

# %%
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")

county_fips = county_fips["county_fips"]

print (f"{len(county_fips.state.unique()) = }")
county_fips = county_fips[county_fips.state.isin(SoI_abb)].copy()
county_fips.drop_duplicates(inplace=True)
county_fips.reset_index(drop=True, inplace=True)
print (f"{len(county_fips.state.unique()) = }")

county_fips.head(2)

# %%
cnty_interest_list = list(county_fips.county_fips.unique())

# %% [markdown]
# ### Inventory

# %%
USDA_data = pd.read_pickle(reOrganized_dir + "USDA_data.sav")
inventory = USDA_data["cattle_inventory"]

# pick only the counties we want
# cattle_inventory = cattle_inventory[cattle_inventory.county_fips.isin(cnty_interest_list)].copy()

print(f"{inventory.data_item.unique() = }")
print(f"{inventory.commodity.unique() = }")
print()
print(f"{len(inventory.state.unique())= }")

inventory.head(2)

# %%
inventory.rename(columns={"cattle_cow_beef_inventory": "inventory"}, inplace=True)

# %%
census_years = sorted(list(inventory.year.unique()))
print(f"{census_years = }")

# pick only useful columns
inv_col_ = "inventory"
inventory = inventory[["year", "county_fips", inv_col_]]

print(f"{len(inventory.county_fips.unique()) = }")
inventory.head(2)

# %% [markdown]
# ### See how many counties and how many data points are incomplete in inventory

# %%
all_cattle_counties = list(inventory.county_fips.unique())
# print(f"{len(all_cattle_counties) = }")
incomplete_counties = {}
for a_cnty_fip in all_cattle_counties:
    curr_cow = inventory[inventory.county_fips == a_cnty_fip].copy()
    missing_yr = [x for x in census_years if not(x in list(curr_cow.year))]
    if (len(missing_yr)>0):
        incomplete_counties[a_cnty_fip] = missing_yr
        
lic = len(incomplete_counties)
la = len(all_cattle_counties)
print ("There are {} incomlete counties out of {} for census years!!!".format(lic, la))

# %%
{key:value for key,value in list(incomplete_counties.items())[0:3]}

# %% [markdown]
# ## NPP exist only after 2001! 
# So let us use subset of cattle inventory from census

# %%
inventory = inventory[inventory.year>=2001]
inventory.reset_index(drop=True, inplace=True)

census_years = sorted(list(inventory.year.unique()))
inventory.head(2)

# %%

# %%
all_cattle_counties = list(inventory.county_fips.unique())
# print(f"{len(all_cattle_counties) = }")
incomplete_counties = {}
for a_cnty_fip in all_cattle_counties:
    curr_cow = inventory[inventory.county_fips == a_cnty_fip].copy()
    missing_yr = [x for x in census_years if not(x in list(curr_cow.year))]
    if (len(missing_yr)>0):
        incomplete_counties[a_cnty_fip] = missing_yr
        
lic = len(incomplete_counties)
la = len(all_cattle_counties)
print ("There are {} incomlete counties out of {} for census years!!!".format(lic, la))

# %% [markdown]
# ## Since there are too many incomlete counties, lets just keep them!

# %% [markdown]
# #### Rangeland area and Herb Ratio

# %%
RA = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")
RA.rename(columns={"fips_id": "county_fips"}, inplace=True)
RA = rc.correct_Mins_county_FIPS(df=RA, col_ = "county_fips")
print (f"{len(RA.county_fips.unique()) = }")
RA = RA[RA.county_fips.isin(cnty_interest_list)]
print (f"{len(RA.county_fips.unique()) = }")
RA.reset_index(drop=True, inplace=True)
RA.head(2)

# %%
cnty_interest_list[:3]

# %%
herb = pd.read_pickle(data_dir_base + "Supriya/Nov30_HerbRatio/county_herb_ratio.sav")
herb = herb["county_herb_ratio"]
herb.head(2)
print (herb.shape)
herb = herb[herb.county_fips.isin(cnty_interest_list)]
print (herb.shape)

herb.dropna(how="any", inplace=True)
print (herb.shape)

herb.reset_index(drop=True, inplace=True)
herb.head(3)

# %%
RA_herb = pd.merge(RA, herb, on=["county_fips"], how="left")
# RA_herb.dropna(how="any", inplace=True)
RA_herb.reset_index(drop=True, inplace=True)
RA_herb.head(2)

# %%
inventory_RA_herb = pd.merge(inventory, RA_herb, on=["county_fips"], how="left")
inventory_RA_herb.head(2)

# %% [markdown]
# ### NPP

# %%
cty_yr_GPP_NPP_prod = pd.read_csv(reOrganized_dir + "county_annual_GPP_NPP_productivity.csv")

cty_yr_GPP_NPP_prod.rename(columns={"county" : "county_fips",
                                    "MODIS_NPP" : "unit_npp"}, inplace=True)
cty_yr_GPP_NPP_prod = rc.correct_Mins_county_FIPS(df=cty_yr_GPP_NPP_prod, col_ = "county_fips")

print (f"{len(cty_yr_GPP_NPP_prod.county_fips.unique()) = }")
cty_yr_GPP_NPP_prod = cty_yr_GPP_NPP_prod[cty_yr_GPP_NPP_prod.county_fips.isin(cnty_interest_list)]
print (f"{len(cty_yr_GPP_NPP_prod.county_fips.unique()) = }")


cty_yr_GPP_NPP_prod.head(5)

# %%
cty_yr_GPP_NPP_prod = pd.merge(cty_yr_GPP_NPP_prod, 
                               RA[["county_fips", "rangeland_acre"]], 
                               on=["county_fips"], how="left")

cty_yr_GPP_NPP_prod = rc.covert_unitNPP_2_total(NPP_df=cty_yr_GPP_NPP_prod, 
                                                npp_unit_col_ = "unit_npp", 
                                                acr_area_col_ = "rangeland_acre", 
                                                npp_area_col_ = "county_total_npp")

cty_yr_GPP_NPP_prod.head(2)

# %%
cty_yr_npp = cty_yr_GPP_NPP_prod[["year", "county_fips", "county_total_npp"]]
cty_yr_npp.dropna(how="any", inplace=True)
cty_yr_npp.reset_index(drop=True, inplace=True)
cty_yr_npp.head(2)

# %%
inventory_RA_herb.head(2)

# %%
inventory_RA_herb_NPP = pd.merge(inventory_RA_herb, cty_yr_npp, 
                                 on=["county_fips", "year"], how="left")

inventory_RA_herb_NPP.head(2)

# %% [markdown]
# ## New Variables compared to old models

# %%
slaughter_Q1 = pd.read_pickle(reOrganized_dir + "slaughter_Q1.sav")
slaughter_Q1 = slaughter_Q1["slaughter_Q1"]
slaughter_Q1.rename(columns={"cattle_on_feed_sale_4_slaughter": "slaughter"}, inplace=True)
slaughter_Q1 = slaughter_Q1[["year", "county_fips", "slaughter"]]
print ("max slaughter sale is [{}]".format(slaughter_Q1.slaughter.max()))
slaughter_Q1.head(2)

# %%
human_population = pd.read_pickle(reOrganized_dir + "human_population.sav")
human_population = human_population["human_population"]
human_population.head(2)

# %%
inventory_RA_herb_NPP_resPop = pd.merge(inventory_RA_herb_NPP, human_population, 
                                        on=["county_fips", "year"], how="left")

inventory_RA_herb_NPP_resPop.head(2)

# %%
USDA_data.keys()

# %%
feed_expense = USDA_data["feed_expense"]
feed_expense = feed_expense[["year", "county_fips", "feed_expense"]]
feed_expense.head(2)

# %%
inventory_RA_herb_NPP_resPop_feedCost = pd.merge(inventory_RA_herb_NPP_resPop, feed_expense, 
                                                 on=["county_fips", "year"], how="left")

inventory_RA_herb_NPP_resPop_feedCost.head(2)

# %%
slaughter_Q1.head(2)

# %%
inventory_RA_herb_NPP_resPop_feedCost_slaughter = pd.merge(inventory_RA_herb_NPP_resPop_feedCost, 
                                                           slaughter_Q1, 
                                                           on=["county_fips", "year"], how="left")

inventory_RA_herb_NPP_resPop_feedCost_slaughter.head(2)

# %%
print (inventory_RA_herb_NPP_resPop_feedCost_slaughter.shape)
all_df = inventory_RA_herb_NPP_resPop_feedCost_slaughter.dropna(how="any", inplace=False)
all_df.reset_index(drop=True, inplace=True)
print (all_df.shape)


all_df.drop(["county_area_acre", "herb_std"], axis="columns", inplace=True)
all_df[all_df.pixel_count == 0]

# %% [markdown]
# ### Add Seasonal Weather Variables
# Seasonal Variables (<--- just for ```ctrl + F```)

# %%
filename = reOrganized_dir + "county_seasonal_temp_ppt_weighted.sav"
seasonal_weather = pd.read_pickle(filename)
print(f"{seasonal_weather.keys() = }")
seasonal_weather = seasonal_weather["seasonal"]
seasonal_weather.head(2)

# %%
SW_vars = ["S1_countyMean_total_precip",
           "S2_countyMean_total_precip",
           "S3_countyMean_total_precip",
           "S4_countyMean_total_precip",
           "S1_countyMean_avg_Tavg",
           "S2_countyMean_avg_Tavg",
           "S3_countyMean_avg_Tavg",
           "S4_countyMean_avg_Tavg"
          ]

for a_col in SW_vars:
    seasonal_weather[a_col] = seasonal_weather[a_col].astype(float)

# %%
all_df = pd.merge(all_df, seasonal_weather, on=["county_fips", "year"], how="left")
all_df.head(2)

# %%
all_df.describe().round(1)

# %%
sorted(all_df.columns)

# %%
controls_noHerb = ["population", "feed_expense", "slaughter", "rangeland_acre"]
controls_wHerb =  ["population", "feed_expense", "slaughter", "rangeland_acre", "herb_avg"]

NPP_control_vars_noHerb= ["county_total_npp"] + controls_noHerb
NPP_control_vars_wHerb = ["county_total_npp"] + controls_wHerb

SW_control_vars_noHerb= SW_vars + controls_noHerb
SW_control_vars_wHerb = SW_vars + controls_wHerb

y_var = "inventory"

# %%
X = all_df[NPP_control_vars_noHerb]
X = sm.add_constant(X)
Y = all_df[y_var].astype(float)
ks = sm.OLS(Y, X)
ks_result =ks.fit()
ks_result.summary()

# %%
del(X, Y, ks, ks_result)

# %% [markdown]
# ## (unbiased) Normalize so ranges are comparable

# %%
all_indp_vars = list(set(NPP_control_vars_noHerb + 
                         NPP_control_vars_wHerb + 
                         SW_control_vars_noHerb + 
                         SW_control_vars_wHerb))
all_indp_vars = sorted(all_indp_vars)
all_indp_vars

# %%
# standard_indp = preprocessing.scale(all_df[explain_vars_herb]) # this is biased
normal_df = (all_df[all_indp_vars] - all_df[all_indp_vars].mean()) / \
                         all_df[all_indp_vars].std(ddof=1)
normal_df.head(2)

# %%
normal_cols = [i + j for i, j in zip(all_indp_vars, ["_normal"] * len(all_indp_vars))]
normal_cols

# %%
all_df[normal_cols] = normal_df
all_df.head(2)

# %%
NPP_control_vars_noHerb_normal = [i + j for i, j in 
                                  zip(NPP_control_vars_noHerb, ["_normal"] * len(NPP_control_vars_noHerb))]

NPP_control_vars_wHerb_normal = [i + j for i, j in 
                                  zip(NPP_control_vars_wHerb, ["_normal"] * len(NPP_control_vars_wHerb))]

SW_control_vars_noHerb_normal = [i + j for i, j in 
                                  zip(SW_control_vars_noHerb, ["_normal"] * len(SW_control_vars_noHerb))]

SW_control_vars_wHerb_normal = [i + j for i, j in 
                                  zip(SW_control_vars_wHerb, ["_normal"] * len(SW_control_vars_wHerb))]

# %%
NPP_control_vars_noHerb_normal

# %%
normal_cols

# %%
all_df.head(2)

# %%
NPP_control_vars_noHerb_normal

# %% [markdown]
# ### NPP and control variables model with herb (normalized)

# %%
X_normal = all_df[NPP_control_vars_wHerb_normal]
X_normal = sm.add_constant(X_normal)
Y = all_df[y_var].astype(float)
ks_normal = sm.OLS(Y, X_normal)
ks_normal_result =ks_normal.fit()
ks_normal_result.summary()

# %% [markdown]
# ### NPP and control variables model No herb (normalized)

# %%
X_normal = all_df[NPP_control_vars_noHerb_normal]
X_normal = sm.add_constant(X_normal)
Y = all_df[y_var].astype(float)
ks_normal = sm.OLS(Y, X_normal)
ks_normal_result =ks_normal.fit()
ks_normal_result.summary()

# %% [markdown]
# # Side-by-sides
#
#   - ```ln(y) = f(NPP)```
#   - ```ln(y) = f(SW)```
#   - ```ln(y) = f(NPP, controls-noHerb)```
#   - ```ln(y) = f(SW, controls-noHerb)```

# %% [markdown]
# ### NPP vs ln(y)

# %%
X = all_df["county_total_npp_normal"]
X = sm.add_constant(X)
Y = np.log(all_df[y_var].astype(float))
ks = sm.OLS(Y, X)
ks_result = ks.fit()
ks_result.summary()

# %%
del(X, ks, ks_result)

# %% [markdown]
# ### SW vs ln(y)

# %%
controls_noHerb_normal_vars = [i + j for i, j in zip(controls_noHerb, ["_normal"] * len(controls_noHerb))]
controls_wHerb_normal_vars = [i + j for i, j in zip(controls_wHerb, ["_normal"] * len(controls_wHerb))]

SW_vars_normal = [x for x in list(SW_control_vars_noHerb_normal) if not(x in list(controls_noHerb_normal_vars))]
SW_vars_normal

# %%
X = all_df[SW_vars_normal]
X = sm.add_constant(X)
Y = np.log(all_df[y_var].astype(float))
ks = sm.OLS(Y, X)
ks_result = ks.fit()
ks_result.summary()

# %%
del(X, ks, ks_result)

# %% [markdown]
# ### NPP and controls (no herb) vs ln(y)

# %%
NPP_control_vars_noHerb_normal

# %%
X = all_df[NPP_control_vars_noHerb_normal]
X = sm.add_constant(X)
Y = np.log(all_df[y_var].astype(float))
ks = sm.OLS(Y, X)
ks_result = ks.fit()
ks_result.summary()

# %%
del(X, ks, ks_result)

# %% [markdown]
# ### SW and controls (no herb) vs ln(y)

# %%
SW_control_vars_noHerb_normal

# %%
X = all_df[SW_control_vars_noHerb_normal]
X = sm.add_constant(X)
Y = np.log(all_df[y_var].astype(float))
ks = sm.OLS(Y, X)
ks_result = ks.fit()
ks_result.summary()

# %%
del(X, ks, ks_result)

# %% [markdown]
# ### NPP and controls (with herb) vs ln(y)

# %%
NPP_control_vars_wHerb_normal

# %%
X = all_df[NPP_control_vars_wHerb_normal]
X = sm.add_constant(X)
Y = np.log(all_df[y_var].astype(float))
ks = sm.OLS(Y, X)
ks_result = ks.fit()
ks_result.summary()

# %%
del(X, ks, ks_result)

# %% [markdown]
# ### SW and controls (with herb) vs ln(y)

# %%
SW_control_vars_wHerb_normal

# %%
X = all_df[SW_control_vars_wHerb_normal]
X = sm.add_constant(X)
Y = np.log(all_df[y_var].astype(float))
ks = sm.OLS(Y, X)
ks_result = ks.fit()
ks_result.summary()

# %% [markdown]
# ## Include lag ```NPP``` in the model
#
# Do we have annual county-level NPP?

# %%

# %%
