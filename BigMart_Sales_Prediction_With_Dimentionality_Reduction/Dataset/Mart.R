install.packages("Amelia")
install.packages("GGally")
install.packages("corrplot")
install.packages("mice")

library(Amelia)    # Has one "missmap" function for finding missing values
library(ggplot2)
library(GGally)    # For the ggpairs function
library(corrplot)
library(mice) # For filling missing values based on the distrubution of other variables as well
####################################################################
# Load the dataset
####################################################################
setwd("C:/Users/prade/Documents/GitHub/university_projects/BigMart_Sales_Prediction_With_Dimentionality_Reduction")
mart <- read.csv("Dataset/Mart_Train.csv", header=TRUE)
head(mart)

# Rename the columns for better viewing
names(mart) <- c("ID", "Weight", "Fat", "Visibility", "Type", "MSRP", "Outlet", "O_Estab", "O_Size", "O_Loc", "O_Type", "Sales")
head(mart)

#####################################################################
# Clean the dataset
#####################################################################

# The data contains many missing values, let's investigate.  
# There is one function for visualizing missing values ... all the missing values
# are in weight ... that helps
missmap(mart)

# Now, let's see if there is any dependence of these missing values on any
# of the other variables.  To do this we will make a temporary copy of the 
# dataset and put a value in for the weight outside of the range
martTmp <- mart
range(martTmp$Weight, na.rm=T)

# This fills in those missing values.  Note that the valid range is 4.5 ... 21.4
# So, replace these values with 50.
martTmp$Weight <- replace(mart$Weight, is.na(mart$Weight) == T, 50)

# Check to see that there are no more missing values
missmap(martTmp)

# Now, let's graph the values ... for the Type, they are spread over the 
# entire set of product types
ggplot(martTmp, aes(x=Type, y=Weight)) + geom_jitter() + coord_flip()

# But we get a different story when look at the Outlet.  It looks like
# only two outlets are systematically not reporting weights
ggplot(martTmp, aes(x=Outlet, y=Weight)) + geom_jitter()

# Let's look at the ID #'s assuming that these identify a unique product
# We cast ID to a number so that the x-axis will be sensible (it won't 
# change the presentation of the graph because the ordering will be the 
# same)
ggplot(martTmp, aes(x=as.numeric(ID), y=Weight)) + geom_point()

# So, let's go into the original dataset and for each missing value, look
# for other stores that have that product ID.  Then we'll set this product
# ID to the average of those (We would actually expect that they are all the
# same)
newMart <- mart
for (i in 1:nrow(newMart))
{
  if (is.na(newMart$Weight[i]))
  {
    prodID = newMart$ID[i]
    matching = newMart[newMart$ID == prodID, ]
    newMart$Weight[i] = mean(matching$Weight, na.rm=T)
  }
}

missmap(newMart)  # Looks like there is only one missing product now ... a vast improvement
ggplot(newMart, aes(x=Type, y=Weight)) + geom_jitter() + coord_flip()
#############################################################################
# Removing incomplete cases
#############################################################################

# Since there are only two products with missing weights now, we discard them
# The "complete.cases" function is designed for this
newMart <- newMart[complete.cases(newMart), ]


# Replace the dataset with the cleaned one
mart <- newMart

###########################################################################
# Dealing with missing values in the O_Size parameter ... warning ... they
# are not "NA's"
###########################################################################

# Unfortunately, there is another kind of missing value ... A missing "blank" 
# O_size parameter
head(mart)

# Let's see what those look like.  Their sales look remarkably like the "Small" 
# category
ggplot(mart, aes(x=O_Size, y=Sales)) + geom_boxplot()

# Looks like they are scattered among Supermarket type 1's and Grocery stores
# But again the distribution looks remarkably like "Small" stores
ggplot(mart, aes(x=O_Size, y=O_Type)) + geom_jitter()

# One last test, looking at the O_Loc.  This doesn't quite match the small 
# pattern.  It looks like the missing values are all in Tier2 and Tier3, and 
# All the other tier2-s  are indeed small, but the tier 3's are mostly meduim
# with a bit of high.  Erring on the size of smaller we could justfy the 
# substitution
#
#  O_Size = Blank && O_Loc == Tier2 --> Small   (most of them are this way)
#  O_Size = Blank && O_Loc == Tier3 --> Medium
ggplot(mart, aes(x=O_Size, y=O_Loc)) + geom_jitter()

# This code does the more complicated substitution, but if you like you can also
# try the simpler substitute of small for the blanks (as it is most of them)
oldMart = mart
mart$O_Size = replace(mart$O_Size, (mart$O_Size == "") & (mart$O_Loc == "Tier 2"), "Small")
mart$O_Size = replace(mart$O_Size, (mart$O_Size == "") & (mart$O_Loc == "Tier 3"), "Medium")
head(mart)

ggplot(mart, aes(x=O_Size, y=O_Loc)) + geom_jitter()

###########################################################################
# Now look at the Visibility variable
###########################################################################

ggplot(mart, aes(x=Type, y=Visibility, fill=Type)) + geom_boxplot()

# Is it reasonable that some of these have a visibility of 0?  That would mean
# perhaps that they aren't on display at all ... but yet they have non-zero sales?

# The following density plot shows a large number of zero visibilities.  The 
# density plot doesn't fall off on the left.
ggplot(mart, aes(x=Visibility)) + geom_density()  

# One option here would be to treat these 0's as missing. Which is what we 
# will do here for the sake of showing another tehcnique for filling missing 
# values.  The usual technique would be to substitute the mean (or median)
# which leaves the mean/median unchange but can DRASTICALLY reduce variance
# and can also cause very strange residual plots (with a large number of identica
# values :)
#
# So, the R "mice" package has a method that will use the overall distribution of 
# the rest of the values relative to the other variables in the set to fill these
# in such a way that the overall distribution will remain the same

# library(mice)
martTmp = mart
martTmp$Visibility = replace(martTmp$Visibility, martTmp$Visibility == 0, NA)
ggplot(martTmp, aes(x=Visibility)) + geom_density()  
md.pattern(martTmp)

# This will take a long time to run
tempData <- mice(martTmp,m=5,maxit=7,meth='pmm',seed=500)
summary(tempData)

###########################################################################
# Now let's look at the correlation matrix
###########################################################################

head(mart)

# Several categorical variables are ordinal, so we change them to numeric
# types for this analysis
levels(mart$O_Size)
levels(mart$O_Loc)
levels(mart$O_Type)

martNumeric = mart[, c(12, 2, 4, 6, 8, 9, 10, 11)]
martNumeric$O_Estab = as.numeric(martNumeric$O_Estab)
martNumeric$O_Loc = as.numeric(martNumeric$O_Loc)
martNumeric$O_Size = as.numeric(martNumeric$O_Size)
martNumeric$O_Type = as.numeric(martNumeric$O_Type)

martNumericClean = martNumeric[complete.cases(martNumeric), ]

# This will fail on any field that has missing values!
c = cor(martNumericClean)
mean(mart$Weight)
corrplot(c, method="ellipse")

# This finds the following major correlations
# 
#     Sales <---> MSRP
#     O_Size <---> O_Loc ... but this is a converted ordinal
#
# Let's take a look at Sales and MSRP

# A worrisome number of outliers, both in the plot vs Type and vs Outlet
# (Notice the two grocery store sales in the sales vs Outlet plot :)
ggplot(mart, aes(x=Type, y=Sales, fill=Type)) + geom_boxplot() + coord_flip()
ggplot(mart, aes(x=Outlet, y=Sales, fill=Outlet)) + geom_boxplot() + coord_flip()

# Fairly stable
ggplot(mart, aes(x=Type, y=MSRP, fill=Type)) + geom_boxplot() + coord_flip()

# We get a distinct linear relationship!  Obviously because Sales = #Units * MSRP!
ggplot(mart, aes(x=MSRP, y=Sales, fill=Type)) + geom_point()

###########################################################################
# Dealing with the Sales/MSRP correlation
###########################################################################

# The big problem is the Salse vs MSRP correlation because MSRP is obviously
# highly corrleated with Sales simply because Sales = #Units * MSRP (unless
# there is a sale, but that will be a % of MSRP).  So let's look at what happens
# when we divide Sales by MSRP to better normalize that value.
ggplot(mart, aes(x=Type, y=Sales/MSRP, fill=Type)) + geom_boxplot() + coord_flip()
ggplot(mart, aes(x=Outlet, y=Sales/MSRP, fill=Outlet)) + geom_boxplot() + coord_flip()

# Since this gives fewer outliers, we will go ahead and roll MSRP into sales
# and create a new parameter of interest 
#
#    stdSales = Sales / MSRP
#
# Note that if we predict stdSales, we will be able to predict Sales by computing
#
#    predSales = predStdSales * MSRP
#
# So this is a valid transform.
mart$stdSales = mart$Sales / mart$MSRP

###########################################################################
# Creating a couple of categories
###########################################################################

# We had a curious pattern in MSRP in one of the previous graphs
ggplot(mart, aes(x=MSRP, y=Sales)) + geom_point()

# Notice that the MSRP's tend to clump into four distinct groups.  We 
# reduce the "bandwidth" of the density calculation to better see the drops
# You can reduce the bandwidth until the top of the curve starts to become 
# chaotic.  We get the minimums here by inspection.
#
# Note ... this is one time when you want many tick marks on the x-axis :)
ggplot(mart, aes(x=MSRP)) + geom_density(bw=2) + scale_x_continuous(breaks=(0:30) * 10) + 
    geom_vline(xintercept=70, color="red") + geom_vline(xintercept=136, color="red") + 
    geom_vline(xintercept=204, color="red")

# So, let's create a new field for the price category, it will be
# 
#     0 < MSRP <= 70  --> "Low"
#    70 < MSRP <= 136 --> "Medium"
#   136 < MSRP <= 204 --> "High"
#   204 < MSRP        --> "VeryHigh"
#
# We start by setting the whole field to "Low" and then we will reset the
# others
mart$PriceCat = factor(rep("Low", nrow(mart)), levels=c("Low", "Medium", "High", "VeryHigh")) 
levels(mart$PriceCat)

# Note that we can do this without compound tests (i.e. (MSRP > 70) && (MSRP < 136))
# Since we will over-write the higher valued categories afterwords.  Note that this
# is HIGHLY dependent on the order that these are executed!
mart$PriceCat[mart$MSRP > 70] = "Medium"
mart$PriceCat[mart$MSRP > 136] = "High"
mart$PriceCat[mart$MSRP > 204] = "VeryHigh"

head(mart)

###########################################################################
# Dealing with many categorical levels
###########################################################################

# Type has 16 levels which can be rather a problem for regression analysis
# as it would mean 15 dummies!
levels(mart$Type)

# Luckily we get a gift here (which you may not get in other sets, but there
# may be other ways to get the same information :) The product "ID" has a 
# leading two-letter code that tells what broad product category the product
# belongs to ... in fact threre are only three of them!
#
#   FD = Food
#   DR = Drinks
#   NC = Non-consumables
unique(substring(mart$ID, 1, 2))

# So let's make a new category "ProductCat"
mart$ProductCat = substring(mart$ID, 1, 2)
head(mart)

# Now, let's look at the other datapoints with respect to this new category.
# Normalized by MSRP this gives a very even set of distributions ... oh well
ggplot(data=mart, aes(x=ProductCat, y=stdSales, fill=ProductCat)) + geom_boxplot()

###########################################################################
# Now create some initial analyses
###########################################################################

library(ggplot2)
library(GGally)
library(car)

names(mart)

# Reorder the fields to get the new Y at the first column
# and then drop a few (like the old "Sales" field, and the 
# "ID" field)
newMart = mart[, c(13, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15)]
newMart = newMart[complete.cases(newMart), ]

head(newMart)
ggpairs(newMart[, -5])  # Type has too many levels for GGally to process

# Don't start off with this! All the categoricals make it incomprehensible
# and nothing will likely be important!  It does tell us that there is 
# about R_squared = .42 out there to capture, but it doesn't tell us HOW!
fit = lm(stdSales ~ ., data=newMart)
summary(fit)

names(newMart)

# Try one with just some numericals?
fit1 = lm(stdSales ~ Weight + Visibility + O_Estab, data=newMart)
summary(fit1)

# How about some of the categoricals ... not much here
fit2 = lm(stdSales ~ ProductCat + PriceCat, data=newMart)
summary(fit2)

library(leaps)
fitFull = lm(stdSales ~ ., data=newMart)
fitNull = lm(stdSales ~ 1, data=newMart)
stepFit = step(fitNull, scope = list(upper=fitFull), data=newMart, direction="both")
summary(stepFit)

# Looks like most of the difference is coming from the outlet ... but not from the outletSize?
# More invesitgation is needed.  So, let's take out "Outlet" and see what we get
fitFull = lm(stdSales ~ . - Outlet, data=newMart)
fitNull = lm(stdSales ~ 1, data=newMart)
stepFit = step(fitNull, scope = list(upper=fitFull), data=newMart, direction="both")
summary(stepFit)

# Suddenly the type of supermarket becomes key.  Remember, all of these dummies are compared to 
# a baseline of "Grocery".  This of course, makes sense but it doesn't really tell us anything 
# about improving sales for a specific type of market except to "Get bigger" ... sigh

#########################################################################################
# Exploring PCA
#########################################################################################

# For this, we will treat the ordinals as numerical, excluding the priceCategory
# because that one is precisely based on another numerical variable that we have

martNumeric = mart[, c(13, 2, 4, 6, 8, 9, 10, 11)]
head(martNumeric)
martNumeric$O_Estab = as.numeric(martNumeric$O_Estab)
martNumeric$O_Loc = as.numeric(martNumeric$O_Loc)
martNumeric$O_Size = as.numeric(martNumeric$O_Size)
martNumeric$O_Type = as.numeric(martNumeric$O_Type)

martNumericClean = martNumeric[complete.cases(martNumeric), ]

head(martNumeric)
pMart = prcomp(martNumericClean[2:8], scale=T)
print(pMart)
summary(pMart)

biplot(pMart)

library(psych)
pMartRot = principal(martNumericClean[2:8], nfactors=3, covar=F)
summary(pMartRot)
print(pMartRot)

source("PCA_Plot.R")
PCA_Plot(pMart)
PCA_Plot_Psyc(pMartRot)

factanal(martNumericClean[2:8], factors=1)
