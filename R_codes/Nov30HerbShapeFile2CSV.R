####
#### This module generates #s in the following link ()
#### https://docs.google.com/document/d/18KX24FkL70_Xhxagwx9EBRWeQmz-Ud-iuTXqnf9YXnk/edit?usp=sharing
####

rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)

##############################################################################

data_dir <- "/Users/hn/Documents/01_research_data/RangeLand/Data/Supriya/Nov30_HerbRatio/"

##############################################################################
# County_State <- readOGR(paste0(data_dir, "County_State.shp"),
#                         layer = "County_State", 
#                         GDAL1_integer64_policy = TRUE)

County_State <- read_sf(paste0(data_dir, "County_State.shp"))

herbRatio <- data.table(data.frame(County_State$GEOID, County_State$Herb_Avgme,
                                   County_State$Herb_SDstd, County_State$Pixelscoun))

setnames(herbRatio, 
         old = c('County_State.GEOID','County_State.Pixelscoun','County_State.Herb_SDstd', 'County_State.Herb_Avgme'), 
         new = c('county_fips','pixel_count','herb_std', 'herb_avg'))

write.csv(herbRatio, file = paste0(data_dir, "herbRatio.csv"), row.names=FALSE)

# nc = st_read(system.file(paste0(data_dir, "County_State.shp"), package="sf"))