# electricity_forecasting

American Electric Power (AEP): estimated energy consumption in Megawatts (MW)
Commonwealth Edison (ComEd): estimated energy consumption in Megawatts (MW)
The Dayton Power and Light Company: estimated energy consumption in Megawatts (MW)
Duke Energy Ohio/Kentucky (DEOK): estimated energy consumption in Megawatts (MW)
Dominion Virginia Power (DOM): estimated energy consumption in Megawatts (MW)
Duquesne Light Co. (DUQ): estimated energy consumption in Megawatts (MW)
East Kentucky Power Cooperative (EKPC): estimated energy consumption in Megawatts (MW)
FirstEnergy (FE): estimated energy consumption in Megawatts (MW)
Northern Illinois Hub (NI): estimated energy consumption in Megawatts (MW)
**PJM East Region: 2001-2018 (PJME): estimated energy consumption in Megawatts (MW)**
PJM West Region: 2001-2018 (PJMW): estimated energy consumption in Megawatts (MW)
PJM Load Combined: 1998-2001 (PJM_Load): estimated energy consumption in Megawatts (MW)
est_hourly.paruqet: Combined All Regions Load in Megawatts (MW): American Electric Power (AEP), Commonwealth Edison (ComEd), Dayton Power and Light Company
pjm_hourly_est: Combined All Regions Load: American Electric Power (AEP), Commonwealth Edison (ComEd), Dayton Power and Light Company

Main goal:
Predict the next 24 hours energy consumption of PJM East Region

I investigated the extreme low values and identified them as real demand drops during Hurricane Sandy in 2012. Since they correspond to real-world events and represent legitimate system behavior, I kept them in the dataset to preserve realism in forecasting.

La serie tiene dependencia local extremadamente fuerte (hora a hora), estacionalidad diaria fuerte y estacionalidad semanal moderada

Baseline 24h MAE: 2300
Como el consumo promedio es 32080, el MAPE es 7,16% lo que implica un baseline fuerte.

External features:
Temperture from Philadelphia Airport.
Heating Degree Days (HDD): it measures how cold is compared to 18ºC: HDD=max(0,18−temp)
Cooling Degree Days (CDD): it measures how hot is compared to 18ºC: CDD=max(0,temp−18)
The relathionship between temperature and demand is not lineal. The demand is high when it's cold and it's hot but is low when the temperature is comfortable. 
HDD and CDD transform that relationship into something that models understand better.

https://electricity-forecast-328829738430.europe-west1.run.app/