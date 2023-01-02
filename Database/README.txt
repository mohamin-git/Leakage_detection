# Metadata for BattLeDIM 2020 Dataset (Website: http://battledim.ucy.ac.cy/)


# File contents description:
"L-TOWN.inp" EPANET Model (INP): 
A model of the network is provided with nominal parameters for all the system elements. 
The nominal base demands for each node are based on average historical metered consumption. 
In general, the difference between the actual and the nominal values for each consumer type (residential and commercial)is less than 10%. 
Weekly demand profiles for two consumer types(residential and commercial) are also provided, however they do not capture the yearly seasonality. Furthermore, the EPANET model parameters may be different from the actual network parameters (e.g., diameters, roughness coefficients), and it is assumed than in general this difference is no greater than 10% of the nominal values.

"L-TOWN_Real.inp" EPANET Model (INP): 
The model of the network used to generate the datasets. 
Contains the real network parameters and consumer demands.
It does not contain any leakages.

"dataset_configuration.yalm" :
Configuration file for generating the Historical (2018) and Evaluation (2019) datasets of the BattLeDIM competition.
Contains:
-file name of network
-dataset start and end times
-leakage information (link ID, start Time, end Time, leak Diameter (m), leak Type, peak Time)
-sensor locations (link or node IDs)

"2018_SCADA_Demands.csv" : 
Part of the Historical Dataset (year 2018) of the BattLeDIM competition.
Water demand flow measurements from 82 Automated Meter Reading devices in the L-Town network. 
Measurement units: Liters per hour (L/h). 
Column headings correspond to Node IDs in the L-Town.inp network file. 

"2018_SCADA_Flows.csv" : 
Part of the Historical Dataset (year 2018) of the BattLeDIM competition.
Flow measurements from 3 flow sensors in the L-Town network.
Measurement units: Cubic meters per hour(m^3/h). 
Column headings correspond to Link IDs in the L-Town.inp network file. 

"2018_SCADA_Levels.csv" :
Part of the Historical Dataset (year 2018) of the BattLeDIM competition.
Tank level measurements from 1 level sensor in the L-Town network
Measurement units: Meters (m). 
Column headings correspond to Tank Node IDs in the L-Town.inp network file. 

"2018_SCADA_Pressures.csv" :
Part of the Historical Dataset (year 2018) of the BattLeDIM competition.
Pressure measurements from 33 pressure sensors in the L-Town network.
Measurement units: Meters (m). 
Column headings correspond to Node IDs in the L-Town.inp network file. 

"2018_SCADA.xlsx" :
Part of the Historical Dataset (year 2018) of the BattLeDIM competition.
Contains all the information of "2018_SCADA_Demands.csv", "2018_SCADA_Flows.csv", "2018_SCADA_Levels.csv" and "2018_SCADA_Pressures.csv" in compact .xlsx form.

"2018_Fixed_Leakages_Report.txt": 
Part of the Historical Dataset (year 2018) of the BattLeDIM competition.
The repair times of pipe bursts that have been fixed in the 2018 dataset are provided.

"2018_Leakages":
Timeseries of all leakages in the Historical Dataset (year 2018) of the BattLeDIM competition.
Measurement units: Cubic meters per hour(m^3/h). 
Column headings correspond to Link IDs in the L-Town.inp network file. 

"2019_SCADA_Demands.csv" : 
Part of the Evaluation Dataset (year 2019) of the BattLeDIM competition.
Water demand flow measurements from 82 Automated Meter Reading devices in the L-Town network. 
Measurement units: Liters per hour (L/h). 
Column headings correspond to Node IDs in the L-Town.inp network file. 

"2019_SCADA_Flows.csv" : 
Part of the Evaluation Dataset (year 2019) of the BattLeDIM competition.
Flow measurements from 3 flow sensors in the L-Town network.
Measurement units: Cubic meters per hour(m^3/h). 
Column headings correspond to Link IDs in the L-Town.inp network file. 

"2019_SCADA_Levels.csv" :
Part of the Evaluation Dataset (year 2019) of the BattLeDIM competition.
Tank level measurements from 1 level sensor in the L-Town network
Measurement units: Meters (m). 
Column headings correspond to Tank Node IDs in the L-Town.inp network file. 

"2019_SCADA_Pressures.csv" :
Part of the Evaluation Dataset (year 2019) of the BattLeDIM competition.
Pressure measurements from 33 pressure sensors in the L-Town network.
Measurement units: Meters (m). 
Column headings correspond to Node IDs in the L-Town.inp network file. 

"2019_SCADA.xlsx" :
Part of the Evaluation Dataset (year 2019) of the BattLeDIM competition.
Contains all the information of "2019_SCADA_Demands.csv", "2019_SCADA_Flows.csv", "2019_SCADA_Levels.csv" and "2019_SCADA_Pressures.csv" in compact .xlsx form.

"2019_Leakages":
Timeseries of all leakages in the Evaluation Dataset (year 2019) of the BattLeDIM competition.
Measurement units: Cubic meters per hour(m^3/h). 
Column headings correspond to Link IDs in the L-Town.inp network file. 



# Acknowledgements
Part of this work has been partially funded by the European Union Horizon 2020 program under Grant Agreement No. 739551 (KIOS CoE), by the Interreg V-A Greece-Cyprus 2014-2020 program, co-financed by the European Union (ERDF) and National Funds of Greece and Cyprus under project SmartWater2020 and by the Deutsche Forschungsgemeinschaft (DFG).
