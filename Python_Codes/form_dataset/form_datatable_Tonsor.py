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
import os, pickle

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

# %%
### Subset seasonal vars to every 5 years that is in the USDA NASS
USDA_years = list(feed_expense.year.unique())
seasonal_5yearLapse = cntyMean_seasonVars_wide[cntyMean_seasonVars_wide.year.isin(USDA_years)].copy()
seasonal_5yearLapse.reset_index(drop=True, inplace=True)
seasonal_5yearLapse.head(2)

# %%
feed_expense.head(2)

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
need_cols = ["year", "county_fips", "wetLand_area", "wetLand_area_cv_(%)"]
season_Feed_CRP = pd.merge(season_Feed, 
                           wetLand_area[need_cols].drop_duplicates(), 
                            on=["year", "county_fips"], how='left')
del(need_cols)

# %%

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
irrigated = AgLand[AgLand.data_item == "AG LAND, IRRIGATED - ACRES"].copy()
irrigated.reset_index(drop=True, inplace=True)
irrigated.head(2)

# %%
if "value" in irrigated.columns:
    irrigated.rename(columns={"value":"irrigated_area", 
                                 "cv_(%)":"irrigated_area_cv_(%)"}, 
                        inplace=True)

print(f"{irrigated.data_item.unique() = }")
irrigated.head(2)

# %%
need_cols = ["year", "county_fips", "irrigated_area", "irrigated_area_cv_(%)"]
season_Feed_CRP_irr = pd.merge(season_Feed_CRP, 
                               irrigated[need_cols].drop_duplicates(), 
                               on=["year", "county_fips"], how='left')
del(need_cols)

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
# pop_2000_2010.head(5)

# %%
# pop_wide.head(5)

# %%
# pop_2000_2010.tail(5)

# %%
# pop_wide.loc[2141*2-6:2141*2]

# %%
# pop_2010_2020.head(5)

# %%
# pop_wide.loc[2141*2:2141*2+5]

# %%
# pop_2010_2020.tail(5)

# %%
# pop_wide.tail(5)

# %%
season_Feed_CRP_irr.head(2)

# %%
pop_wide.head(2)

# %%
season_Feed_CRP_pop = pd.merge(season_Feed_CRP, pop_wide, 
                               on=["year", "county_fips"], how='left')


# %%
print (f"{season_Feed_CRP.shape = }")
print (f"{season_Feed_CRP_pop.shape = }")
season_Feed_CRP_pop.head(5)

# %%

# %%
