# A Random Forest Approach to Tornado-Day Forecasting in the Deep South

Authors: Brandon Weart, Daniel Wefer

### Data Download
The data for this project can be located on the Amazon Web Service (AWS) cloud server: https://registry.opendata.aws/noaa-gefs-reforecast/

And at: https://www.spc.noaa.gov/wcm/


These Python packages are required to run the code in this repository. 
* Pandas 
* Matplotlib 
* NumPy 
* Cartopy 
* Scikit-learn


### Data Analysis

Jupyter notebooks containing the code used to not only train the model, but produce the figures shown in this README can be found in this repository under the Notebooks folder. 


### Introduction

Forecasting tornadoes with machine learning has become a major staple in the atmospheric science community (Gensini et al. 2020; 2021), and will continue to be ever so important as we advance our understanding of machine learning, as well as develop new techniques to forecast tornadoes. Tornadoes are notoriously difficult to forecast due to their still poorly understood dynamics, but a general subset of ingredients exists that can be used to varying degrees of success to predict environments conducive to tornadoes. For this project, a random forest model was utilized to attempt to find correlations with ingredients to different calibers of tornado days in the Deep South: days with no tornadoes, days with no significant tornadoes, and days with significant tornadoes. 


### Data and Methods
#### Working with the data
Though we did not explicitly define thresholds for tornado days, we were able to extract days fitting thresholds defining non-tornado days, tornado days, and significant tornado days.

_Table 1: Tornado-day definitions_
| Category (label)        | Definition                                                                      |
|:------------------------|:--------------------------------------------------------------------------------|
| No-Tornado Day   (0)    | A day that does not exist within the SPC tornado report archive                 |
| Tornado Day      (1)    | A day when one or more tornadoes **below** EF-2 rating occurred                 |
| Significant Tornado Day (2) | A day when one or more tornadoes **at or above** EF-2 rating occurred           |


The temporal search period was from 2001 to 2019. Each of the days pulled was assigned labels based on the criteria defined (Table 1). In total, 259, 256, and 197 days were found for labels 0,1,2, respectively. This does not reflect the total number of days within the period being searched, as using the total number of days would create an unbalanced dataset that would prove detrimental to the model's performance, as a disproportionately large amount of 0s exist in the dataset since tornadoes are rare events. It is important to note that this search only gathered reports within out domain (Fig. 1).




After extracting the days from the SPC dataset, we utilized the Global Ensemble Forecast System (GEFS) reforecast dataset to represent the atmosphere for a given date. For each date, we used a subset of 17 variables arbitrarily determined to be associated with tornado-prone environments (Table 2).

_Table 2: Environmental variables extracted from GEFS at each grid cell_
| Abbreviation | Full Name                                           |
|--------------|-----------------------------------------------------|
| u10          | 10 m zonal (east–west) wind speed                   |
| v10          | 10 m meridional (north–south) wind speed            |
| t2m          | 2 m air temperature                                 |
| td2m         | 2 m dewpoint temperature                            |
| cprec        | Convective precipitation                            |
| srh500m      | 0–500 m storm-relative helicity                     |
| srh1km       | 0–1 km storm-relative helicity                      |
| srh3km       | 0–3 km storm-relative helicity                      |
| scp          | Supercell composite parameter                       |
| stp          | Significant tornado parameter                       |
| vtp          | Violent Tornado parameter                           |
| lcl          | Lifting condensation level                          |
| shear6km     | Bulk wind shear (surface–6 km)                      |
| mslp         | Mean sea level pressure                             |
| pwat         | Precipitable water                                  |
| sbcape       | Surface-based convective available potential energy |
| sbcin        | Surface-based convective inhibition                 |



This dataset is gridded and has multiple timesteps for each day. As we are using a random forest model, we need to convert this data into a tabular format. To do this, we extracted the max value over the time dimension at each grid cell, then took the sum, median, and max of those variables and appended them to a CSV for each date. There is an interesting correlation between the distribution of each variable for each label type (Fig. 2).

![Boxplots](Images/boxplots.png)

Before the model can be trained on the data, it needed to be split up into three different subsets: training, validation, testing. Training data is utilized to initally train the model and usually encompasses the largest portion of the data. The validation data is used as a sort of pseudo-test, and allows you to see the results of the model, and further tweak the hyperparameters of the model as the user sees fit. Scikit-learn has a built in function to take care of splitting the data into training and testing, so a workaround was utilized to create the validation dataset from a temporary subset. Utilizing the scikit-learn train-test split function (Scikit-learn 2025a), the data was first split into train and "temp", the temporary subset mentioned. The temporary subset contained 30% of the data. It was further split into validation and testing with a second iteration of the function, splitting the test to be 66% of the temp. The data was split 70%, 10%, 20% for train, val, test, respectively. 


### Model configuration and results

To initialize the random forest model, we utilized Scikit-learns RandomForestClassifer function (Scikit-learn 2025b). After modifying the hyperparameters, the best model configuration was found to
