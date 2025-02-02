##################################################################################################################################################
# Author: Pradeep Sathyamurthy, Daniel Glownia
# Team Mates: Ashrita, Meghana, Daniel
# Guiding Professor: Dr. Eli T Brown
# Course: CSC-465
# Project: Final Course Project for CSC-465, visualizing DIVVY dataset
# Part-1: Gathering right shape file for Divvy Dataset and fortifying DIVVY dataset with it
# Part-2: Plotting Choropleth for Chicago City based on DP Capacity and when each station went live
# Date Created: 21-Oct-2016
# Date Last Modified: 20-Nov-2016
##################################################################################################################################################

# Setting up the project directory
setwd("D:/Courses/CSC465 - Tableau - Data Visualization/Trails/Choroplath")
install.packages("maptools")
install.packages("rgeos")
install.packages("Cairo")
install.packages("proto")
install.packages("ggmap")
install.packages("scales")
install.packages("RColorBrewer")

require(ggplot2)
require(rgeos)
require(maptools)
require(Cairo)
require(ggmap)
require(scales)
require(RColorBrewer)

set.seed(8000)

# Reading the shape file
chicago_neighbor.shp <- readShapeSpatial("geo_export_f83f4fda-cb0e-47e2-9ffa-859f2b78e325.shp")

# Checking its class
class(chicago_neighbor.shp)

# Checking the names associated with shape file
names(chicago_neighbor.shp)

# Checking the valus into it
print(chicago_neighbor.shp$objectid)
print(chicago_neighbor.shp$zip)

##create (or input) data to plot on map
num.neighbor <- length(chicago_neighbor.shp$objectid)

# Getting DIVVY dataset in 
mydata_chi <- read.csv("Divvy_Stations_2016_Q1Q2_Modified.csv", stringsAsFactors = FALSE)
head(mydata_chi)
print(mydata_chi$shp_id)
print(chicago_neighbor.shp$objectid)

# fortify shape file to get into dataframe, one of the piece of code which took considerably long time
neigh.shp.f <- ggplot2::fortify(chicago_neighbor.shp, region = "objectid")
class(neigh.shp.f)
head(neigh.shp.f)

#merge with coefficients and reorder
merge.shp.coef3 <- merge(neigh.shp.f, mydata_chi, by="id", all.x=TRUE)
final.chi.plot <- merge.shp.coef3[order(merge.shp.coef3$order), ] 

# ggplot for plotting choropleth using geom_polygon
# plottiing Choropleth filled with dpcapacity
c1 <- ggplot() +
    geom_polygon(data = final.chi.plot, 
                 aes(x = long, y = lat, group = group, fill = dpcapacity), 
                 color = "black", size = 0.25) + 
    coord_map()+
    scale_fill_distiller(name="Docking Capacity", palette = "YlGnBu",trans="reverse", breaks = pretty_breaks(n = 5))+
    theme_nothing(legend = TRUE)+
    labs(title="DP Capacity in Chicago")

# Histogram to assist the choropleth plotted for visualization
c2 <- ggplot(data = mydata_chi,aes(dpcapacity)) + geom_histogram(binwidth = 7)+ 
        labs(x="Docking Capacity", y="Count of Divvy Stations") + 
        ggtitle("Distribution of Docking Capacity in Divvy Station")


# saving the file to local working directory
ggsave(c1, file = "final_divvy_choroplath.png", width = 6, height = 4.5, type = "cairo-png")
ggsave(c2, file = "Divvy_DP_Distribution.png", width = 6, height = 4.5, type = "cairo-png")


    

 