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
from datetime import datetime
import os, pickle, sys

# %%
# import geopandas
# A = geopandas.read_file("/Users/hn/Desktop/amin/shapefile.shp")
# A.head(2)

# %% [markdown]
# According to Google Map the coordinate ```30.59375N 88.40625W``` is in Alabama. According to Bhupi's file it
# is in Alabama by state but in Mississipi by ```fips```. That place is on the border of the two states. I will go by fips.

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_dir_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"
seasonal_dir = reOrganized_dir + "seasonal_variables/02_merged_mean_over_county/" 

# %%
# file_name = "countyMean_seasonalVars_wFips.csv"
# countyMean_seasonalVars = pd.read_csv(seasonal_dir + file_name)
# print (f"{len(countyMean_seasonalVars.state.unique())=}")

# # round numbers
#  countyMean_seasonalVars = countyMean_seasonalVars.round(decimals=2)
# countyMean_seasonalVars.head(2)

# %%
# file_name = "countyMean_seasonalVars_wFips.sav"
# countyMean_seasonalVars = pickle.load(open(seasonal_dir + file_name, "rb"))
# del(file_name)
# countyMean_seasonalVars = countyMean_seasonalVars["countyMean_seasonalVars_wFips"]
# countyMean_seasonalVars.head(2)

# %%
Min_FIPS = pd.read_csv("/Users/hn/Documents/01_research_data/Sid/Analog/parameters/Min_counties.csv")
Min_FIPS = Min_FIPS[["state", "county", "fips"]]
Min_FIPS.drop_duplicates(inplace=True)
Min_FIPS.reset_index(drop=True, inplace=True)
Min_FIPS.head(2)

# %%
Min_FIPS[Min_FIPS.state == "AL"].sort_values(by=['county'])

# %%
Bhupi = pd.read_csv(param_dir + "Bhupi_25states_clean.csv")
Bhupi["SC"] = Bhupi.state + "-" + Bhupi.county
Bhupi.head(2)

# %%
SoI = ["Alabama", "Arkansas", 
       "California", "Colorado", 
       "Florida", "Georgia",
       "Idaho", "Illinois", 
       "Iowa", "Kansas", 
       "Kentucky", "Louisiana", 
       "Mississippi", "Missouri", 
       "Montana", "Nebraska", 
       "New Mexico", "North Dakota", 
       "Oklahoma", "Oregon", 
       "South Dakota", "Tennessee", 
       "Texas", "Virginia", "Wyoming"]

# %%
cntyMean_seasonVars_wide = pickle.load(open(seasonal_dir + "wide_seasonal_vars_cntyMean_wFips.sav", "rb"))
cntyMean_seasonVars_wide = cntyMean_seasonVars_wide["wide_seasonal_vars_cntyMean_wFips"]

cntyMean_seasonVars_wide.sort_values(by=['state', 'county', 'year'], inplace=True)
cntyMean_seasonVars_wide.reset_index(drop=True, inplace=True)

# %%
USDA_data = pickle.load(open(reOrganized_dir + "USDA_data.sav", "rb"))

feed_expense = USDA_data["feed_expense"]
AgLand = USDA_data["AgLand"]
wetLand_area = USDA_data["wetLand_area"]
totalBeefCowInv = USDA_data["totalBeefCowInv"]
# FarmOperation = USDA_data["FarmOperation"] # not needed. create by NASS guy.

# %%
cntyMean_seasonVars_wide = cntyMean_seasonVars_wide.round(decimals=2)
cntyMean_seasonVars_wide.head(2)

# %%
feed_expense[(feed_expense.state=="Alabama") & (feed_expense.county=="Baldwin")]

# %%
feed_expense[(feed_expense.state=="Alabama") & (feed_expense.county=="Washington")]

# %%
print (f"{len(feed_expense.state.unique())=}")
feed_expense.data_item.unique()

# %%
feed_expense.rename(columns={"value":"feed_expense", 
                             "cv_(%)":"feed_expense_cv_(%)"}, inplace=True)
feed_expense.head(2)

# %%
feed_expense = feed_expense[feed_expense.state.isin(SoI)].copy()

print (feed_expense.shape)
print(len(feed_expense.state.unique()))
print(len(feed_expense.county.unique()))
print(len(feed_expense.year.unique()))

feed_expense.head(2)

# %%
### Subset seasonal vars to every 5 years that is in the USDA NASS
USDA_years = list(feed_expense.year.unique())
seasonal_5yearLapse = cntyMean_seasonVars_wide[cntyMean_seasonVars_wide.year.isin(USDA_years)].copy()
seasonal_5yearLapse.reset_index(drop=True, inplace=True)
seasonal_5yearLapse.head(2)

# %%
#
# Merge seasonal variables and feed expenses.
#
need_cols = ["year", "county_fips", "feed_expense", "feed_expense_cv_(%)"]
season_Feed = pd.merge(seasonal_5yearLapse, 
                       feed_expense[need_cols].drop_duplicates(), 
                       on=["year", "county_fips"], how='left')

del(need_cols)
print (f"{seasonal_5yearLapse.shape = }")
print (f"{feed_expense.shape = }")
print (f"{season_Feed.shape = }")
season_Feed.head(2)

# %%
feed_expense_FIPS_yr = feed_expense["county_fips"].astype(str)  + "_" + \
                          feed_expense["year"].astype(str)

seasonal_5yearLapse_FIPS_yr = seasonal_5yearLapse["county_fips"].astype(str)  + "_" + \
                                 seasonal_5yearLapse["year"].astype(str)

A = [x for x in list(seasonal_5yearLapse_FIPS_yr) if not (x in list(feed_expense_FIPS_yr))]
print (f"{len(A) = }")
A[:10]

# %%
feed_expense[feed_expense.county_fips=="06075"]

# %%
seasonal_5yearLapse[seasonal_5yearLapse.county_fips=="06075"]

# %%
season_Feed[season_Feed.county_fips=="06075"]

# %% [markdown]
# ### CRP
# ```Wetland``` file:
#
# Acres enrolled in Conservation Reserve, Wetlands Reserve, Farmable Wetlands, or Conservation Reserve Enhancement Programs, 1997-2017:\\
#
# https://quickstats.nass.usda.gov/#3A734A89-9829-3674-AFC6-C4764DF7B728

# %%
if "value" in wetLand_area.columns:
    wetLand_area.rename(columns={"value":"wetLand_area", 
                                 "cv_(%)":"wetLand_area_cv_(%)"}, 
                        inplace=True)

print(f"{wetLand_area.data_item.unique() = }")
wetLand_area.head(2)

# %%
wetLand_area_25state = wetLand_area[wetLand_area.state.isin(SoI)]
print (wetLand_area_25state.shape)
print(len(wetLand_area_25state.state.unique()))
print(len(wetLand_area_25state.county.unique()))
print(len(wetLand_area_25state.year.unique()))

# %%
need_cols = ["year", "county_fips", "wetLand_area", "wetLand_area_cv_(%)"]
season_Feed_CRP = pd.merge(season_Feed, 
                           wetLand_area[need_cols].drop_duplicates(), 
                            on=["year", "county_fips"], how='left')
del(need_cols)

# %%
wetLand_area.year.unique()

# %%
print (f"{season_Feed.shape = }")
print (f"{wetLand_area.shape = }")
print (f"{season_Feed_CRP.shape = }")

# %% [markdown]
# ### Irrigated Acre
#
# ```AgLand```. Irrigated acres and total land in farms by county, 1997-2017.
#
# https://quickstats.nass.usda.gov/#B2688D70-61AC-3E14-AA15-11882355E95E
#
# Ask for **Kirti**'s input.

# %%
print ("AgLand.data_item are {}".format(list(AgLand.data_item.unique())))
print ()
AgLand.head(2)

# %%
# wetLand_area_25state = wetLand_area[wetLand_area.state.isin(SoI)]
print (AgLand.shape)
print(len(AgLand.state.unique()))
print(len(AgLand.county.unique()))
print(len(AgLand.year.unique()))

# %%
AgLand_25state = AgLand[AgLand.state.isin(SoI)]
print (AgLand_25state.shape)
print(len(AgLand_25state.state.unique()))
print(len(AgLand_25state.county.unique()))
print(len(AgLand_25state.year.unique()))

# %%

# %% [markdown]
# # TEST
#
# Test and see what are the ```data_item```s here. Do they reflect percentages in WA counties?
#
# We need to add those in terms of percentage not absolute values.

# %%
# COI = ['Adams', 'Benton', 'Franklin', 'Grant', 'Walla Walla', 'Yakima']

# WSDA_SF_data_dir = "/Users/hn/Documents/01_research_data/00_shapeFiles/01_shapefiles_data_part_not_filtered/"
# WSDA_SF_data_dir_2 = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"

# WSDA_2018 = pd.read_csv(WSDA_SF_data_dir + "WSDA_DataTable_2018.csv")
# print (WSDA_2018.county.unique())
# WSDA_2018.head(2)

# %%
# WSDA_2018 = WSDA_2018[WSDA_2018.county.isin(COI)]
# len(WSDA_2018.ID.unique())

# %%
# all_eastern = pd.read_csv(WSDA_SF_data_dir_2 + "all_SF_data_concatenated.csv")

# all_eastern["SF_year"] = all_eastern.ID.str.split("_", expand=True)[3]
# all_eastern["SF_year"] = all_eastern["SF_year"].astype(int)

# all_eastern_2018 = all_eastern[all_eastern.SF_year == 2018].copy()
# print (len(all_eastern_2018.ID.unique()))
# print (all_eastern_2018.county.unique())
# all_eastern.head(2)

# %%
# all_eastern_total_Acr_perCounty = all_eastern[["ExctAcr", "county"]].groupby(["county"]).sum().reset_index()
# all_eastern_2018_total_Acr_perCounty = all_eastern_2018[["ExctAcr", "county"]].groupby(
#                                                            ["county"]).sum().reset_index()
# WSDA_2018_total_Acr_perCounty = WSDA_2018[["ExctAcr", "county"]].groupby(["county"]).sum().reset_index()

# %%
# sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
# import NASA_core as nc
# all_eastern_irr = nc.filter_out_nonIrrigated(all_eastern)
# all_eastern_2018_irr = nc.filter_out_nonIrrigated(all_eastern_2018)

# print (f"{all_eastern.shape = }")
# print (f"{all_eastern_irr.shape = }")
# all_eastern_irr.head(2)

# all_eastern_irr_Acr_perCounty = all_eastern_irr[["ExctAcr", "county"]].groupby(["county"]).sum().reset_index()
# all_eastern_2018_irr_Acr_perCounty = all_eastern_2018_irr[["ExctAcr", "county"]].groupby(["county"]
#                                                                                         ).sum().reset_index()

# all_eastern_irr_Acr_perCounty.rename(columns={"ExctAcr":"irr_area"}, inplace=True)
# all_eastern_2018_irr_Acr_perCounty.rename(columns={"ExctAcr":"irr_area"}, inplace=True)

# %%
# WSDA_areas_df = pd.merge(all_eastern_total_Acr_perCounty, 
#                          all_eastern_irr_Acr_perCounty, 
#                          on=["county"], how='left')
# WSDA_areas_df["irr_as_perc"] = (WSDA_areas_df.irr_area / WSDA_areas_df.ExctAcr)*100
# WSDA_areas_df.iloc[:, 1:] = WSDA_areas_df.iloc[:, 1:].round(2)
# WSDA_areas_df

# WSDA_2018_irr = nc.filter_out_nonIrrigated(WSDA_2018)
# WSDA_2018_irr_Acr_perCounty = WSDA_2018_irr[["ExctAcr", "county"]].groupby(["county"]).sum().reset_index()
# WSDA_2018_irr_Acr_perCounty.rename(columns={"ExctAcr":"irr_area"}, inplace=True)
# WSDA_2018_irr_Acr_perCounty

# WSDA_2018_areas_df = pd.merge(WSDA_2018_total_Acr_perCounty, 
#                               WSDA_2018_irr_Acr_perCounty, 
#                               on=["county"], how='left')
# WSDA_2018_areas_df["irr_as_perc"] = (WSDA_2018_areas_df.irr_area / WSDA_2018_areas_df.ExctAcr)*100
# WSDA_2018_areas_df.iloc[:, 1:] = WSDA_2018_areas_df.iloc[:, 1:].round(2)
# WSDA_2018_areas_df

# %%

# %%
# AgLand_WA = AgLand[AgLand.state == "Washington"].copy()
# AgLand_WA.rename(columns={"value":"area"}, inplace=True)
# AgLand_WA.area = AgLand_WA.area.replace(',','', regex=True)
# AgLand_WA.area = AgLand_WA.area.astype(int)
# AgLand_WA = AgLand_WA[AgLand_WA.county.isin(COI)]
# AgLand_irrigated_WA = AgLand_WA[AgLand_WA.data_item == "AG LAND, IRRIGATED - ACRES"].copy()
# AgLand_irrigated_WA.reset_index(drop=True, inplace=True)

# AgLand_FarmOper_WA = AgLand_WA[AgLand_WA.data_item == "FARM OPERATIONS - ACRES OPERATED"].copy()
# AgLand_FarmOper_WA.reset_index(drop=True, inplace=True)

# print (AgLand_irrigated_WA.shape)
# AgLand_irrigated_WA.head(2)

# %%
# AgLand_FarmOper_WA.head(2)

# %%
# AgLand_WA_FarmOper_acr_perCounty = AgLand_FarmOper_WA[["county", "area", "year"]].groupby(
#                                                     ["county", "year"]).sum().reset_index()
# AgLand_WA_FarmOper_acr_perCounty.head(2)

# %%
# AgLand_WA_irr_acr_perCounty = AgLand_irrigated_WA[["county", "area", "year"]].groupby(
#                                                      ["county", "year"]).sum().reset_index()
# AgLand_WA_irr_acr_perCounty.rename(columns={"area":"irr_area"}, inplace=True)

# %%
# Agland_areas_df = pd.merge(AgLand_WA_total_acr_perCounty, 
#                            AgLand_WA_irr_acr_perCounty, 
#                            on=["county", "year"], how='left')
# Agland_areas_df["irr_as_perc"] = (Agland_areas_df.irr_area / Agland_areas_df.area)*100
# Agland_areas_df.iloc[:, 1:] = Agland_areas_df.iloc[:, 1:].round(2)
# Agland_areas_df.head(10)

# %%
# Agland_areas_df2 = pd.merge(AgLand_WA_FarmOper_acr_perCounty, 
#                             AgLand_WA_irr_acr_perCounty, 
#                             on=["county", "year"], how='left')
# Agland_areas_df2["irr_as_perc"] = (Agland_areas_df2.irr_area / Agland_areas_df2.area)*100
# Agland_areas_df2.iloc[:, 1:] = Agland_areas_df2.iloc[:, 1:].round(2)
# Agland_areas_df2.head(10)

# %%
AgLand.rename(columns={"value":"area"}, inplace=True)

AgLand = AgLand[AgLand.area != ' (D)'].copy()
AgLand.area = AgLand.area.replace(',','', regex=True)
AgLand.area = AgLand.area.astype(int)

AgLand_FarmOper = AgLand[AgLand.data_item == "FARM OPERATIONS - ACRES OPERATED"].copy()
AgLand_FarmOper.reset_index(drop=True, inplace=True)

AgLand_irrigated = AgLand[AgLand.data_item == "AG LAND, IRRIGATED - ACRES"].copy()
AgLand_irrigated.reset_index(drop=True, inplace=True)
AgLand_irrigated.head(2)

# %%

# %%
# if "value" in wetLand_area.columns:
#     wetLand_area.rename(columns={"value":"irrigated_area", 
#                                  "cv_(%)":"irrigated_area_cv_(%)"}, 
#                         inplace=True)

# print(f"{wetLand_area.data_item.unique() = }")
# AgLand_irrigated.head(2)

# %%
# I do not know why I have "irrigated.columns" below.
# GitHub said wetLand. So, I fixed it above.
if "value" in AgLand_irrigated.columns:
    AgLand_irrigated.rename(columns={"value":"irrigated_area", 
                                     "cv_(%)":"irrigated_area_cv_(%)"}, 
                        inplace=True)

print(f"{AgLand_irrigated.data_item.unique() = }")
AgLand_irrigated.head(2)

# %%
AgLand_FarmOper_acr_perCounty = AgLand_FarmOper[["state", "county", "area", "year", "county_fips"]].groupby(
                                                    ["state", "county", "year", "county_fips"]).sum().reset_index()
AgLand_FarmOper_acr_perCounty.head(2)

AgLand_irr_acr_perCounty = AgLand_irrigated[["state", "county", "area", "year", "county_fips"]].groupby(
                                                     ["state", "county", "year", "county_fips"]).sum().reset_index()
AgLand_irr_acr_perCounty.rename(columns={"area":"irr_area"}, inplace=True)


# %%
Agland_areas_df = pd.merge(AgLand_FarmOper_acr_perCounty, 
                            AgLand_irr_acr_perCounty, 
                            on=["state", "county", "year", "county_fips"], how='left')
Agland_areas_df["irr_as_perc"] = (Agland_areas_df.irr_area / Agland_areas_df.area)*100
Agland_areas_df.iloc[:, 1:] = Agland_areas_df.iloc[:, 1:].round(2)
Agland_areas_df.head(5)

# %%
Agland_areas_df[Agland_areas_df.state=="Washington"].head(5)

# %%
season_Feed_CRP.head(2)

# %%
Agland_areas_df.head(2)

# %%
season_Feed_CRP.head(2)

# %%
need_cols = ["state", "county", "year", "county_fips", "irr_as_perc"]
season_Feed_CRP_irr = pd.merge(season_Feed_CRP, 
                               Agland_areas_df[need_cols].drop_duplicates(), 
                               on=["year", "county_fips", "state", "county"], how='left')
del(need_cols)

season_Feed_CRP_irr.head(2)

# %%

# %% [markdown]
# ### County Population
#
# This is population of people. I do not know how this is relevant. Each farm would send beef outside of the county, no?

# %%
population_file_names = [x for x in os.listdir(census_population_dir) if x.startswith("z")]
population_file_names

# %%
pop_2000_2010 = pd.read_csv(census_population_dir + "z_2000_2010_co-est00int-tot.csv", encoding= 'unicode_escape')
pop_2010_2020 = pd.read_csv(census_population_dir + "z_2010-2020-co-est2020.csv", encoding= 'unicode_escape')


pop_2000_2010.drop(['SUMLEV', 'REGION', "DIVISION", ], axis=1, inplace=True)
pop_2000_2010.rename(columns={"STATE": "state_fip", 
                              "COUNTY": "cnty_fip",
                              "STNAME":"state", 
                              "CTYNAME":"county"}, inplace=True)

pop_2010_2020.drop(['SUMLEV', 'REGION', "DIVISION", ], axis=1, inplace=True)
pop_2010_2020.rename(columns={"STATE": "state_fip", 
                              "COUNTY": "cnty_fip",
                              "STNAME":"state", 
                              "CTYNAME":"county"}, inplace=True)

# %%
pop_2000_2010.head(2)

# %%
pop_2010_2020.head(2)

# %%
pop_2000_2010.state_fip = pop_2000_2010.state_fip.astype(str)
pop_2000_2010.cnty_fip = pop_2000_2010.cnty_fip.astype(str)

for idx in pop_2000_2010.index:
    if len(pop_2000_2010.loc[idx, "state_fip"]) == 1:
        pop_2000_2010.loc[idx, "state_fip"] = "0" + pop_2000_2010.loc[idx, "state_fip"]
    
    col = "cnty_fip"
    if len(pop_2000_2010.loc[idx, col]) == 1:
        pop_2000_2010.loc[idx, col] = "00" + pop_2000_2010.loc[idx, col]
    elif len(pop_2000_2010.loc[idx, col]) == 2:
        pop_2000_2010.loc[idx, col] = "0" + pop_2000_2010.loc[idx, col]
        
pop_2000_2010.head(2)

# %%
pop_2010_2020.state_fip = pop_2010_2020.state_fip.astype(str)
pop_2010_2020.cnty_fip = pop_2010_2020.cnty_fip.astype(str)

for idx in pop_2010_2020.index:
    if len(pop_2010_2020.loc[idx, "state_fip"]) == 1:
        pop_2010_2020.loc[idx, "state_fip"] = "0" + pop_2010_2020.loc[idx, "state_fip"]
    
    col = "cnty_fip"
    if len(pop_2010_2020.loc[idx, col]) == 1:
        pop_2010_2020.loc[idx, col] = "00" + pop_2010_2020.loc[idx, col]
    elif len(pop_2010_2020.loc[idx, col]) == 2:
        pop_2010_2020.loc[idx, col] = "0" + pop_2010_2020.loc[idx, col]

pop_2010_2020.head(2)

# %% [markdown]
# ### ```cnty_fip == 000```
#
# presentes all state. i.e. sum of all counties. Drop them.

# %%
pop_2000_2010.drop(pop_2000_2010[pop_2000_2010.cnty_fip == "000"].index, inplace=True)
pop_2010_2020.drop(pop_2010_2020[pop_2010_2020.cnty_fip == "000"].index, inplace=True)

# %%
pop_2000_2010["county_fips"] = pop_2000_2010["state_fip"] + pop_2000_2010["cnty_fip"]
pop_2010_2020["county_fips"] = pop_2010_2020["state_fip"] + pop_2010_2020["cnty_fip"]

pop_2000_2010.drop(['state_fip', 'cnty_fip'], axis=1, inplace=True)
pop_2010_2020.drop(['state_fip', 'cnty_fip'], axis=1, inplace=True)

pop_2000_2010.head(2)

# %%
pop_2010_2020.head(2)

# %%
season_Feed_CRP_irr.year.unique()

# %%
pop_2000_2010 = pop_2000_2010[pop_2000_2010.state.isin(SoI)]
pop_2010_2020 = pop_2010_2020[pop_2010_2020.state.isin(SoI)]

pop_2000_2010.reset_index(drop=True, inplace=True)
pop_2010_2020.reset_index(drop=True, inplace=True)

# %%

# %%
need_cols_2000_2010 = ["county_fips", "POPESTIMATE2002", "POPESTIMATE2007"]
need_cols_2010_2020 = ["county_fips", "POPESTIMATE2012", "POPESTIMATE2017"]

# %%
pop_2000_2010 = pop_2000_2010[need_cols_2000_2010]
pop_2010_2020 = pop_2010_2020[need_cols_2010_2020]

# %%
pop_2000_2010.head(2)

# %%
pop_2010_2020.head(2)

# %%
print (f"{len(pop_2000_2010) = }")
print (f"{len(pop_2010_2020) = }")

# %%
A = [x for x in pop_2010_2020.county_fips if not(x in list(pop_2000_2010.county_fips))]
A

# %%
A = [x for x in pop_2000_2010.county_fips if not(x in list(pop_2010_2020.county_fips))]
A

# %%
print ("46113" in list(season_Feed_CRP_irr.county_fips))
print ("51515" in list(season_Feed_CRP_irr.county_fips))
print ("46102" in list(season_Feed_CRP_irr.county_fips))

# %%
A = [x for x in pop_2000_2010.county_fips if not(x in list(season_Feed_CRP_irr.county_fips))]
A

# %%
A = [x for x in pop_2010_2020.county_fips if not(x in list(season_Feed_CRP_irr.county_fips))]
A

# %%
L = len(pop_2000_2010) + len(pop_2010_2020)
pop_wide = pd.DataFrame(columns=["county_fips", "year", "population"], index=range(L*2))

pop_wide.county_fips = "666"
pop_wide.year = 666
pop_wide.population = 666

years_2000_2010 = [2002, 2007]*len(pop_2000_2010)
years_2010_2020 = [2012, 2017]*len(pop_2010_2020)
pop_wide.year = years_2000_2010 + years_2010_2020

pop_wide.head(5)

# %%
wide_pointer = 0 

for idx in pop_2000_2010.index:
    pop_wide.loc[wide_pointer, "county_fips"] = pop_2000_2010.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer, "population"] = pop_2000_2010.loc[idx, "POPESTIMATE2002"]
    
    pop_wide.loc[wide_pointer+1, "county_fips"] = pop_2000_2010.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer+1, "population"] = pop_2000_2010.loc[idx, "POPESTIMATE2007"]
    wide_pointer += 2
    
for idx in pop_2010_2020.index:
    pop_wide.loc[wide_pointer, "county_fips"] = pop_2010_2020.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer, "population"] = pop_2010_2020.loc[idx, "POPESTIMATE2012"]
    
    pop_wide.loc[wide_pointer+1, "county_fips"] = pop_2010_2020.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer+1, "population"] = pop_2010_2020.loc[idx, "POPESTIMATE2017"]
    wide_pointer += 2

# %%
season_Feed_CRP_irr.head(2)

# %%
pop_wide.head(2)

# %%
# AgLand_25state = AgLand[AgLand.state.isin(SoI)]
print (pop_wide.shape)
print(len(pop_wide.county_fips.unique()))
# print(len(pop_wide.county.unique()))
# print(len(pop_wide.year.unique()))

# %%
print (season_Feed_CRP_irr.shape)
print(len(season_Feed_CRP_irr.county_fips.unique()))

# %%
season_Feed_CRP_irr_pop = pd.merge(season_Feed_CRP_irr, pop_wide, 
                                   on=["year", "county_fips"], how='left')

season_Feed_CRP_irr_pop.head(2)

# %%
print (f"{season_Feed_CRP_irr.shape = }")
print (f"{season_Feed_CRP_irr_pop.shape = }")
season_Feed_CRP_irr_pop.head(5)

# %% [markdown]
# ### Beef Cow Inventory (heads)
#
#  - Total Beef Cow inventory: https://quickstats.nass.usda.gov/#ADD6AB04-62EF-3DBF-9F83-5C23977C6DC7
#  - Inventory of Beef Cows: https://quickstats.nass.usda.gov/#B2688D70-61AC-3E14-AA15-11882355E95E

# %%
# totalBeefCowInv = pd.read_csv(USDA_data_dir + "totalBeefCowInv.csv")
# totalBeefCowInv.head(2)

# %%
print (len(totalBeefCowInv.state.unique()))
print (len(totalBeefCowInv.county.unique()))
print (len(totalBeefCowInv.year.unique()))

totalBeefCowInv.head(2)

# %%
totalBeefCowInv.rename(columns={"value":"total_beefCowInv", 
                             "cv_(%)":"total_beefCowInv_(%)"}, inplace=True)
totalBeefCowInv.head(2)

# %%
totalBeefCowInv = totalBeefCowInv[totalBeefCowInv.state.isin(SoI)].copy()

print (totalBeefCowInv.shape)
print(len(totalBeefCowInv.state.unique()))
print(len(totalBeefCowInv.county.unique()))
print(len(totalBeefCowInv.year.unique()))

totalBeefCowInv.head(2)

# %%
need_cols = ["year", "county_fips", "total_beefCowInv", "total_beefCowInv_(%)"]
season_Feed_CRP_irr_pop_beef = pd.merge(season_Feed_CRP_irr_pop, 
                                        totalBeefCowInv[need_cols].drop_duplicates(), 
                                        on=["year", "county_fips"], how='left')

# %%
season_Feed_CRP_irr_pop_beef.head(2)

# %%

# %%
