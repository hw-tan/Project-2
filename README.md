# Project 2: Ames Housing Data and Kaggle Challenge
## Excutive Summary

In this notebook we seek to use regression to predict housing prices. There are many factors that affect home prices, the more common factors are the size, the quality and the age of the property.

In this notebook we use a dataset from Kaggle, the goal of the dataset is to predict the sales price of home given the features of the home. As home prices are continuous, this is a regression problem. As such we select features in the dataset with Exploratory Data Analysis to fit into an OLS regression model. We then compare the results with different types of regularisation (Ridge and Lasso).

The best model with the lowest Root Mean Squared Error (RMSE) will be selected to fit the training data predict a test set provided by Kaggle. This will be uploaded in Kaggle Competitition and will be evaluated by the RMSE.

A good model that captures enough variables and charateristics of the house will be useful in many ways. For prediction, it serves to determine the fair value of the property given its characteristics. For inference, we can get a flavour as to what variables affect the price of houses.

## Problem Statement


What combination of parameters and hyperparameters should be used in a regression model to best predict housing prices?

## Data importing and cleaning
The dataset used in this study is from the Ames Iowa Boston Housing Dataset. It contains information on residential properties sold in Ames from 2006 to 2010.

The approach taken was to rename columns, check for null values and the data type of the features.

### Data Dictionary

Below is the data dictionary from the raw data.

| Feature | Type | Dataset | Description |  |
|-|-|-|-|-|
| id | int64 | train | Order ID |  |
| pid | int64 | train | Parcel identification number  - can   be used with city web site for parcel review.  |  |
| ms_subclass | int64 | train | Identifies the type of dwelling involved in the sale.	 |  |
| ms_zoning | object | train | Identifies the general zoning classification of the sale. |  |
| lot_frontage | float64 | train | Linear feet of street connected to property |  |
| lot_area | int64 | train | Lot size in square feet |  |
| street | object | train | Type of road access to property |  |
| alley | object | train | Type of alley access to property |  |
| lot_shape | object | train | General shape of property |  |
| land_contour | object | train | Flatness of the property |  |
| utilities | object | train | Type of utilities available |  |
| lot_config | object | train | Lot configuration |  |
| land_slope | object | train | Slope of property |  |
| neighborhood | object | train | Physical locations within Ames city limits (map available) |  |
| condition_1 | object | train | Proximity to various conditions |  |
| condition_2 | object | train | Proximity to various conditions (if more than one is present) |  |
| bldg_type | object | train | Type of dwelling |  |
| house_style | object | train | Style of dwelling |  |
| overall_qual | int64 | train | Rates the overall material and finish of the house |  |
| overall_cond | int64 | train | Rates the overall condition of the house |  |
| year_built | int64 | train | Original construction date |  |
| year_remod/add | int64 | train | Remodel date (same as construction date if no remodeling or additions) |  |
| roof_style | object | train | Type of roof |  |
| roof_matl | object | train | Roof material |  |
| exterior_1st | object | train | Exterior covering on house |  |
| exterior_2nd | object | train | Exterior covering on house (if more than one material) |  |
| mas_vnr_type | object | train | Masonry veneer type |  |
| mas_vnr_area | float64 | train | Masonry veneer area in square feet |  |
| exter_qual | object | train | Evaluates the quality of the material on the exterior  |  |
| exter_cond | object | train | Evaluates the present condition of the material on the exterior |  |
| foundation | object | train | Type of foundation |  |
| bsmt_qual | object | train | Evaluates the height of the basement |  |
| bsmt_cond | object | train | Evaluates the general condition of the basement |  |
| bsmt_exposure | object | train | Refers to walkout or garden level walls |  |
| bsmtfin_type_1 | object | train | Rating of basement finished area |  |
| bsmtfin_sf_1 | float64 | train | Type 1 finished square feet |  |
| bsmtfin_type_2 | object | train | Rating of basement finished area (if multiple types) |  |
| bsmtfin_sf_2 | float64 | train | Type 2 finished square feet |  |
| bsmt_unf_sf | float64 | train | Unfinished square feet of basement area |  |
| total_bsmt_sf | float64 | train | Total square feet of basement area |  |
| heating | object | train | Type of heating |  |
| heating_qc | object | train | Heating quality and condition |  |
| central_air | object | train | Central air conditioning |  |
| electrical | object | train | Electrical system |  |
| 1st_flr_sf | int64 | train | First Floor square feet |  |
| 2nd_flr_sf | int64 | train | Second floor square feet |  |
| low_qual_fin_sf | int64 | train | Low quality finished square feet (all floors) |  |
| gr_liv_area | int64 | train | Above grade (ground) living area square feet |  |
| bsmt_full_bath | float64 | train | Basement full bathrooms |  |
| bsmt_half_bath | float64 | train | Basement half bathrooms |  |
| full_bath | int64 | train | Full bathrooms above grade |  |
| half_bath | int64 | train | Half baths above grade |  |
| bedroom_abvgr | int64 | train | Bedrooms above grade (does NOT include basement bedrooms) |  |
| kitchen_abvgr | int64 | train | Kitchens above grade |  |
| kitchen_qual | object | train | Kitchen quality |  |
| totrms_abvgrd | int64 | train | Total rooms above grade (does not include bathrooms) |  |
| functional | object | train | Home functionality (Assume typical unless deductions are warranted) |  |
| fireplaces | int64 | train | Number of fireplaces |  |
| fireplace_qu | object | train | Fireplace quality |  |
| garage_type | object | train | Garage location |  |
| garage_yr_blt | float64 | train | Year garage was built |  |
| garage_finish | object | train | Interior finish of the garage |  |
| garage_cars | float64 | train | Size of garage in car capacity |  |
| garage_area | float64 | train | Size of garage in square feet |  |
| garage_qual | object | train | Garage quality |  |
| garage_cond | object | train | Garage condition |  |
| paved_drive | object | train | Paved driveway |  |
| wood_deck_sf | int64 | train | Wood deck area in square feet |  |
| open_porch_sf | int64 | train | Open porch area in square feet |  |
| enclosed_porch | int64 | train | Enclosed porch area in square feet |  |
| 3ssn_porch | int64 | train | Three season porch area in square feet |  |
| screen_porch | int64 | train | Screen porch area in square feet |  |
| pool_area | int64 | train | Pool area in square feet |  |
| pool_qc | object | train | Pool quality |  |
| fence | object | train | Fence quality |  |
| misc_feature | object | train | Miscellaneous feature not covered in other categories |  |
| misc_val | int64 | train | $Value of miscellaneous feature |  |
| mo_sold | int64 | train | Month Sold (MM) |  |
| yr_sold | int64 | train | Year Sold (YYYY) |  |
| sale_type | object | train | Type of sale |  |
| saleprice | int64 | train | Sale price $$ |  |

## Exploratory Data Analysis

The purpose of the exploratory data analysis is to select features for the regression model that will best predict the sales price of the property. We will be looking for variables that have a strong and clear relationship with saleprice.

As there are 81 variables in the dataset, it is not time efficient or meaningful to explain each of the variables. Instead, in this section, we will analyse the variables that has a clear relationship with saleprice and has some level of variation in the variable. After analysing the selected variables, we will breifly explain why some variables were rejected.

From filtering the available variables, we selected the following features for consideration for the model:

| Feature | Type | From | Category | Description |
|-|-|-|-|-|
|irreg_shape | Dummy | created | Quality | 1 if lot shape is irregular else 0 |
| inside_lot | Dummy | created | Quality | 1 if lot configuration is inside lot else 0 |
| one_floor | Dummy | created | Size | 1 if house style is 1 story or 1.5 story with 2nd level unfinished |
| good_qual | Dummy | created | Quality | 1 if overall quality of house is above 7 else 0 |
| property_age | int64 | created | Age | Age of property when sold |
| have_mas | Dummy | created | Quality | 1 if property has masonry veneer else 0 |
| good_ext | Dummy | created | Quality | 1 if External quality is good or excellent else 0 |
| total_bsmt_sf | float64 | dataset | Size | Total square feet of basement area |
| 1st_flr_sf | Int64 | dataset | Size | First Floor square feet |
| 2nd_flr_sf | Int64 | dataset | Size | Second floor square feet |
| gr_liv_area | Int64 | dataset | Size | Above grade living area square feet |
| total_bath | Int64 | dataset | Size | Total full bath in the property |
| good_kitchen | Dummy | created | Quality | 1 if kitchen quality is good or excellent |
| totrms_abvgrd | Int64 | dataset | Size | Total rooms above grade |


## Modeling

Our main task is to predict the sale price of properties in Ames Iowa. We have selected features that have a linear-like relationship with sale price and will likely provide some predictive powers. In this section, we will run through the iterative process of fitting a model. 

1) Select variables

2) Tune model with hyperparameters

3) K-fold cross validation

In this regression analysis, we will be using regular OLS and regularised regression algorithms (Lasso and Ridge). The features were transformed to the 2nd order polynomials with interaction term. The following is the results of the model when cross validated in a train test split:

|Model|Features|mean RMSE from CV|
|---|---|---|
|OLS|regular|36642.33|
|Ridge|regular|36505.57|
|Lasso|regular|36423.65|
|OLS|degree-2 polynomial|77917.89|
|Ridge|degree-2 polynomial|31632.67|
|Lasso|degree-2 polynomial|31497.68|

## Kaggle submission

The model was used to predict unseen test data provided by Kaggle. The predictions got a RMSE of 28945.83121.


