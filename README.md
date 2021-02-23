# OLAM_validation

Programs and scripts used for the OLAM model validation

The programs and scripts contained here wee elaborated for performing analysis for the above mentioned research article:

Ocean-Land Atmosphere Model (OLAM) performance for major extreme meteorological events near the coastal region of southern Brazil, submitted for Climate Research, in revision 2020.

All the original data related can be obtained at https://data.4tu.nl/articles/dataset/Processed_Data_of_Extreme_Events_for_the_Coastal_Region_of_Southern_Brazil/12721367/2

Atuhors: Souza & Ramos da Silva

## Usage

In order to run the scripts exactly as they are provided here, you need to download the data and separe it in the following folders:

* GPM_TRMM_netcdf
* MERRA_netcdf
* OLAM_netcdf
* Station_data

Additionally, you should create the following folders and sub-folders for saving the figures:

*animations
  * slp_and_wind
  * wind_prec
* figures
  * accprec
  * accprec_time_evo
  * lat_sens_heat
  * satellite
  * slp_wind
  * station

## Data

Data is separated into events. There are 12 events in total and the corresponding dates can be checking in the original manuscript/article or by opening the NetCDFs files and assessing the time dimension

Reanalysis data contained here (and the link for download) is listed below:

	* Modern-Era Retrospective analysis for Research and Applications, Version 2 (MERRA-2, https://disc.gsfc.nasa.gov/datasets/M2I3NPASM_5.12.4/summary)
	* The Tropical Rainfall Measuring Mission (TRMM, https://disc.gsfc.nasa.gov/datasets/TRMM_3B42RT_7/summary?keywords=TRMM_3B42RT_7)
	* Global Precipitation Measurement (GPM, https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_06/summary?keywords=%22IMERG%20final%22)

The OLAM configuration files can be found at https://data.4tu.nl/articles/dataset/Processed_Data_of_Extreme_Events_for_the_Coastal_Region_of_Southern_Brazil/12721367/2
