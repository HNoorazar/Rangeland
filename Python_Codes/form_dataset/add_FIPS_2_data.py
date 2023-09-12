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

# %%
dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"

NASS_dir = dir_base + "NASS_downloads/"
census_dir = dir_base + "census/"


# %%
feed_expense = pd.read_csv(NASS_dir + "feed_expense.csv")

# %%
sorted(feed_expense.columns)

# %%
feed_expense["County ANSI"].unique()

# %%
feed_expense["State ANSI"].unique()

# %%
feed_expense[feed_expense["State ANSI"]==1]

# %%
