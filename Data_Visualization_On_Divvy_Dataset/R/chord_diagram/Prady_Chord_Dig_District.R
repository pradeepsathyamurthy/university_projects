install.packages("dplyr")
install.packages("circlize")
require(dplyr)
require(circlize)

# Create Fake Flight Information in a table
setwd("D:/Courses/CSC465 - Tableau - Data Visualization/Trails/Network Plot")
data.divvy_04 <- read.csv("Divvy_Edges_Sub_Dt_Dataset.csv",na.strings = NULL, stringsAsFactors = FALSE)

set.seed(2400000)
divvy_subscriber <- data.divvy_04[which(data.divvy_04$Type=="Subscriber"),]
divvy_Customer <- data.divvy_04[which(data.divvy_04$Type=="Customer"),]

origin_subs <- divvy_subscriber$Source
dest_subs <- divvy_subscriber$Target
origin_cust <- divvy_Customer$Source
dest_cust <- divvy_Customer$Target

df_subs = data.frame(origin_subs, dest_subs)
df_cust = data.frame(origin_cust, dest_cust)

# Create a Binary Matrix Based on mydf
matrix_subs <- data.matrix(as.data.frame.matrix(table(df_subs)))
matrix_cust<- data.matrix(as.data.frame.matrix(table(df_cust)))

# create the objects you want to link from to in your diagram
from_subs <- rownames(matrix_subs)
to_subs <- colnames(matrix_subs)
from_cust <- rownames(matrix_cust)
to_cust <- colnames(matrix_cust)

# Create Diagram by suppling the matrix 
par(mar = c(1, 1, 1, 1))
#?chordDiagram
#chordDiagram(matrix_subs, transparency=0 ,order = sort(union(from_subs, to_subs)), directional = TRUE)
#chordDiagram(matrix_cust, transparency=0 ,order = sort(union(from_cust, to_cust)), directional = TRUE)
#circos.clear()


grid.col = c("#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628","#f781bf")

grid.col2 = c("#e41a1c","#1b9e77","#7570b3","#e7298a","#a6761d","#d95f02","#66a61e","#e6ab02")
#"#d95f02","#1b9e77","#7570b3","#e7298a","#666666","#a6761d","#66a61e","#e6ab02"
#"#666666","#1b9e77","#7570b3","#e7298a","#d95f02","#a6761d","#66a61e","#e6ab02"


# Diff Colour rendering
#chordDiagram(matrix_cust, grid.col = grid.col, row.col = c("#FF000080", "#00FF0010", "#0000FF10"))

#chordDiagram(matrix_cust, grid.col = grid.col, transparency = 0)
# Below is good
#col_mat = rand_color(length(matrix_cust), transparency = 0.5)
#dim(col_mat) = dim(matrix_cust) # to make sure it is a matrix
#chordDiagram(matrix_cust, grid.col = grid.col, col = col_mat)

#chordDiagram(matrix_cust, grid.col = grid.col, column.col = 1:6)

# Graphic settings for link
#chordDiagram(matrix_cust, grid.col = grid.col, link.lwd = 2, link.lty = 2, link.border = "black")

# Highlight links by colour
#col_mat[matrix_cust < 12] = "#00000000"
#chordDiagram(matrix_cust, grid.col = grid.col, col = col_mat)

# Sort Link on Sector, this is very clear
#chordDiagram(matrix_cust, grid.col = grid.col, link.border = 1,
#             text(-0.9, 0.9, "A", cex = 1.5))

# Direction connection
chordDiagram(matrix_cust, grid.col = grid.col2,annotationTrack = "grid", self.link = 2, link.border = 0)
circos.clear()
# Organization of track
#chordDiagram(matrix_cust, grid.col = grid.col, annotationTrack = c("name", "grid"),
#             annotationTrackHeight = c(0.03, 0.01))

# Customize sector labels
#chordDiagram(matrix_cust, grid.col = grid.col, annotationTrack = "grid",
#             preAllocateTracks = list(track.height = 0.3))
# we go back to the first track and customize sector labels
circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
    xlim = get.cell.meta.data("xlim")
    ylim = get.cell.meta.data("ylim")
    sector.name = get.cell.meta.data("sector.index")
    circos.text(mean(xlim), ylim[1], sector.name, facing = "clockwise",
                niceFacing = TRUE, adj = c(0, 0.25))
}, bg.border = NA) # here set bg.border to NA is important


