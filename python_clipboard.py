 def get_system_costs(wacc_model, benchmark_lcoe, technology):

    # Create a storage array
    system_costs = xr.full_like(wacc_model.land_mapping, np.nan)
    
    # Extract dataframe that contains the wind and solar penetration
    renewable_penetration = wacc_model.calculated_data[['index', 'Wind_Penetration_2023', 'Solar_Penetration_2023']]
    if technology == "Solar":
        renewable_penetration = renewable_penetration[['index', 'Solar_Penetration_2023']].rename(columns={"Solar_Penetration_2023":"Penetration"})
    elif technology == "Wind":
        renewable_penetration = renewable_penetration[['index', 'Wind_Penetration_2023']].rename(columns={"Wind_Penetration_2023":"Penetration"})                                
    
    # Loop over country indexes
    for index in np.arange(1, 251, 1): 
        # Calculate Penetration
        penetration = renewable_penetration.loc[renewable_penetration['index'] == index]
        if penetration.empty:
            system_costs = xr.where(wacc_model.land_mapping == index, 0, system_costs)
            continue
            
        penetration_value = penetration['Penetration'].values[0] / 100
        
        # Calculate Profile Costs
        profile = 60.4 * penetration_value + 3.35
        
        # Calculate Balancing Costs
        balancing = 7.28 * penetration_value + 2.215
        
        # Calculate system costs
        system_costs = xr.where(wacc_model.land_mapping == index, profile + balancing, system_costs)
    
    # Calculate benchmark value
    benchmark_values = xr.where(np.isnan(system_costs), np.nan, (benchmark_lcoe - system_costs ) ) / 1000
    
    return benchmark_values

def calculate_required_waccs(wacc_model, data, lcoe=None, benchmark_lcoe=None, technology=None):

    # Drop existing data
    data = data.drop_vars('Required_WACC')
    
    # Extract key figures from the data
    latitudes = data.latitude.values
    longitudes = data.longitude.values
    annual_electricity_production = data['electricity_production'].sel(year=2022)
    initial_lcoe = data['Calculated_LCOE']

    # Calculate annual costs
    annual_costs = data['Calculated_OPEX']
    capital_costs = data['Calculated_CAPEX']

    
    # Get LCOE 
    if lcoe is None:
        lcoe = get_system_costs(wacc_model, benchmark_lcoe, technology)
        lcoe = xr.where(np.isnan(initial_lcoe), np.nan, lcoe)
        data['LCOE'] = lcoe
        print(np.nanmean(lcoe))
    else:
        lcoe = lcoe / 1000
        print(np.nanmean(lcoe))

    # Calculate discount factor at each location
    # Ensure that the denominator is not zero or negative
    valid_mask = (annual_electricity_production * lcoe  - annual_costs) > 0

    # Apply the calculation only where valid
    discount_factor = xr.where(
        np.isnan(lcoe) | ~valid_mask,
        np.nan,
        capital_costs / ((annual_electricity_production * lcoe) - annual_costs)
    )

    data['Discount_Factor'] = discount_factor
    
    # Create array of discount factor to WACC values and round discount factor
    discount_rates = np.linspace(0, 0.25, 1001)
    discount_factors_array = wacc_model.calculate_discount_factor(discount_rates)
    xdata = discount_rates
    ydata = discount_factors_array

    # Calculate curve fit
    ylog_data = np.log(ydata)
    curve_fit = np.polyfit(xdata, ylog_data, 2)
    y = np.exp(curve_fit[2]) * np.exp(curve_fit[1]*xdata) * np.exp(curve_fit[0]*xdata**2)


    # Create interpolator
    interpolator = interp1d(ydata, xdata, kind='nearest', bounds_error=False, fill_value=(np.nan, 9.99))

    # Use rounded discount factors to calculate WACC values 
    estimated_waccs = interpolator(discount_factor)*100
    estimated_waccs = xr.where(discount_factor < 0, 999, estimated_waccs)
    estimated_waccs = xr.where(discount_factor < 0, np.nan, estimated_waccs)
    wacc_da = xr.DataArray(estimated_waccs, coords={"latitude": latitudes, "longitude":longitudes})
    data['Benchmark_WACC'] = xr.where(np.isnan(initial_lcoe)==True, np.nan, wacc_da)

    return data


solar_test_results = calculate_required_waccs(wacc_model, wacc_model.solar_results, lcoe=73)
solar_benchmark_results = calculate_required_waccs(wacc_model, wacc_model.solar_results, benchmark_lcoe=73, technology="Solar")


wacc_model.plot_data_shading(solar_benchmark_results['Benchmark_WACC'], solar_test_results.latitude, solar_test_results.longitude, tick_values = [0, 5, 10, 15, 20]) 
wacc_model.plot_data_shading(solar_test_results['Benchmark_WACC'], solar_test_results.latitude, solar_test_results.longitude, tick_values = [0, 5, 10, 15, 20])
print(np.nanmax(solar_test_results['Benchmark_WACC']-solar_benchmark_results['Benchmark_WACC']))

def get_concessionality(wacc_model, data, concessional_leverage, technology):
    
    # Convert data in netcdf to pandas dataframe
    merged_dataset = xr.Dataset({
    "Required_WACC": data,
    "index": wacc_model.land_mapping
})
    working_dataframe = merged_dataset.to_dataframe()
    
    # Store latitude and longitude
    
    # Import the existing costs of commercial debt, equity and the modelled debt share
    if technology == "Wind":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Onshore_Wind_WACC_2023', 'Wind_Cost_Debt_2023', 'Wind_Cost_Equity_2023']].rename(columns={"Onshore_Wind_WACC_2023":"WACC_2023", "Wind_Cost_Debt_2023": "Debt_Cost_2023", "Wind_Cost_Equity_2023":"Equity_Cost_2023"})
    elif technology == "Solar":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Solar_WACC_2023', 'Solar_Cost_Debt_2023', 'Solar_Cost_Equity_2023']].rename(columns={"Solar_WACC_2023":"WACC_2023", "Solar_Debt_2023": "Debt_Cost_2023", "Solar_Equity_2023":"Equity_Cost_2023"})
    financing_terms = pd.merge(working_dataframe, financing_values, how="left", on="index") 
    print(financing_terms)
    
    # Calculate the debt / equity ratio
    financing_terms['Debt_Equity'] = financing_terms['Debt_Share_2023'] / (100 -  financing_terms['Debt_Share_2023'])
    financing_terms['Conc_Equity_Share'] = 1 / ( (1 + financing_terms['Debt_Equity']) / concessional_leverage + (1 + financing_terms['Debt_Equity']))
    financing_terms['Commercial_Debt_Share'] = financing_terms['Debt_Equity'] * financing_terms['Conc_Equity_Share']
    financing_terms['Concessional_Debt_Share'] = 1 - financing_terms['Commercial_Debt_Share'] - financing_terms['Conc_Equity_Share']
    
    # Calculate shares of final
    financing_terms['Concessional_Rate'] =  (financing_terms['Required_WACC'] - financing_terms['Conc_Equity_Share'] * financing_terms['Equity_Cost_2023'] - financing_terms['Commercial_Debt_Share'] * financing_terms['Debt_Cost_2023'])    / financing_terms['Concessional_Debt_Share']
    financing_terms['Concessionality'] = financing_terms['Debt_Cost_2023'] - financing_terms['Concessional_Rate']
    
    # Create a check
    financing_terms['WACC_Concessional'] = financing_terms['Conc_Equity_Share'] * financing_terms['Equity_Cost_2023'] 
    + financing_terms['Commercial_Debt_Share'] * financing_terms['Debt_Cost_2023'] 
    + financing_terms['Concessional_Rate'] * financing_terms['Concessional_Debt_Share']
    
    return financing_terms

def get_concessionality_v2(wacc_model, data, concessional_leverage, technology):
    
    # Convert data in netcdf to pandas dataframe
    merged_dataset = xr.Dataset({
    "Required_WACC": data,
    "index": wacc_model.land_mapping
})
    working_dataframe = merged_dataset.to_dataframe().reset_index()
    
    # Import the existing costs of commercial debt, equity and the modelled debt share
    if technology == "Wind":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Onshore_Wind_WACC_2023', 'Wind_Cost_Debt_2023', 'Wind_Cost_Equity_2023']].rename(columns={"Onshore_Wind_WACC_2023":"WACC_2023", "Wind_Cost_Debt_2023": "Debt_Cost_2023", "Wind_Cost_Equity_2023":"Equity_Cost_2023"})
    elif technology == "Solar":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Solar_WACC_2023', 'Solar_Cost_Debt_2023', 'Solar_Cost_Equity_2023']].rename(columns={"Solar_WACC_2023":"WACC_2023", "Solar_Debt_2023": "Debt_Cost_2023", "Solar_Equity_2023":"Equity_Cost_2023"})
    financing_terms = pd.merge(working_dataframe, financing_values, how="left", on="index") 
    print(financing_terms)
    
    # Calculate the debt / concessional debt / equity shares
    financing_terms['Equity_Share'] = (100 - financing_terms['Debt_Share_2023']) / 100
    financing_terms['Concessional_Debt_Share'] = 1 / (1 + concessional_leverage)
    financing_terms['Commercial_Debt_Share'] = 1 - financing_terms['Equity_Share'] - financing_terms['Concessional_Debt_Share']
    
    # Calculate shares of final
    financing_terms['Equity_Contribution'] = financing_terms['Equity_Share'] * financing_terms['Equity_Cost_2023']
    financing_terms['Debt_Contribution'] = financing_terms['Commercial_Debt_Share'] * financing_terms['Debt_Cost_2023']
    financing_terms['Concessional_Rate'] =  (financing_terms['Required_WACC'] - financing_terms['Equity_Contribution']  - financing_terms['Debt_Contribution'])    / financing_terms['Concessional_Debt_Share']
    financing_terms['Concessionality'] = financing_terms['Debt_Cost_2023'] - financing_terms['Concessional_Rate']
    
    # Create a check
    financing_terms['WACC_Concessional'] =  financing_terms['Equity_Contribution'] + financing_terms['Debt_Contribution'] + financing_terms['Concessional_Rate'] * financing_terms['Concessional_Debt_Share']
    
    # Address inequalities
    financing_terms.loc[financing_terms['Concessional_Rate'] < 0, "Concessional_Rate"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 999, "Concessional_Rate"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 111, "Concessional_Rate"] = 111
    
    # Convert back to netcdf
    financing_terms = financing_terms.set_index(["latitude", "longitude"])
    processed_data = financing_terms.to_xarray()
    
    return financing_terms, processed_data
        
concessional_calcs, processed_data = get_concessionality_v2(wacc_model, wind_benchmark_wacc, 4, "Wind")
wacc_model.plot_data_shading(processed_data['Concessional_Rate'], processed_data.latitude, processed_data.longitude, special_value = 999, hatch_label = "", special_value_2 = 111, hatch_label_2="", tick_values = [0, 5, 10, 15, 20, 25, 30])



def get_concessionality_v2(wacc_model, data, concessional_leverage, technology):
    
    # Convert data in netcdf to pandas dataframe
    merged_dataset = xr.Dataset({
    "Required_WACC": data,
    "index": wacc_model.land_mapping
})
    working_dataframe = merged_dataset.to_dataframe().reset_index()
    
    # Import the existing costs of commercial debt, equity and the modelled debt share
    if technology == "Wind":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Onshore_Wind_WACC_2023', 'Wind_Cost_Debt_2023', 'Wind_Cost_Equity_2023']].rename(columns={"Onshore_Wind_WACC_2023":"WACC_2023", "Wind_Cost_Debt_2023": "Debt_Cost_2023", "Wind_Cost_Equity_2023":"Equity_Cost_2023"})
    elif technology == "Solar":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Solar_WACC_2023', 'Solar_Cost_Debt_2023', 'Solar_Cost_Equity_2023']].rename(columns={"Solar_WACC_2023":"WACC_2023", "Solar_Debt_2023": "Debt_Cost_2023", "Solar_Equity_2023":"Equity_Cost_2023"})
    financing_terms = pd.merge(working_dataframe, financing_values, how="left", on="index") 
    print(financing_terms)
    
    # Calculate the debt / concessional debt / equity shares
    financing_terms['Equity_Share'] = (100 - financing_terms['Debt_Share_2023']) / 100
    financing_terms['Concessional_Debt_Share'] = 1 / (1 + concessional_leverage)
    financing_terms['Commercial_Debt_Share'] = 1 - financing_terms['Equity_Share'] - financing_terms['Concessional_Debt_Share']
    
    # Calculate shares of final
    financing_terms['Equity_Contribution'] = financing_terms['Equity_Share'] * financing_terms['Equity_Cost_2023']
    financing_terms['Debt_Contribution'] = financing_terms['Commercial_Debt_Share'] * financing_terms['Debt_Cost_2023']
    financing_terms['Concessional_Rate'] =  (financing_terms['Required_WACC'] - financing_terms['Equity_Contribution']  - financing_terms['Debt_Contribution'])    / financing_terms['Concessional_Debt_Share']
    financing_terms['Concessionality'] = financing_terms['Debt_Cost_2023'] - financing_terms['Concessional_Rate']
    
    # Create a check
    financing_terms['WACC_Concessional'] =  financing_terms['Equity_Contribution'] + financing_terms['Debt_Contribution'] + financing_terms['Concessional_Rate'] * financing_terms['Concessional_Debt_Share']
    
    # Address inequalities
    financing_terms.loc[financing_terms['Concessional_Rate'] < 0, "Concessional_Rate"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 999, "Concessional_Rate"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 111, "Concessional_Rate"] = 111
    
    # Convert back to netcdf
    financing_terms = financing_terms.set_index(["latitude", "longitude"])
    processed_data = financing_terms.to_xarray()
    
    return financing_terms, processed_data






def get_concessionality_v2(wacc_model, data, concessional_leverage, technology):
    
    # Convert data in netcdf to pandas dataframe
    merged_dataset = xr.Dataset({
    "Required_WACC": data,
    "index": wacc_model.land_mapping
})
    working_dataframe = merged_dataset.to_dataframe().reset_index()
    
    # Import the existing costs of commercial debt, equity and the modelled debt share
    if technology == "Wind":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Onshore_Wind_WACC_2023', 'Wind_Cost_Debt_2023', 'Wind_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Onshore_Wind_WACC_2023":"WACC_2023", "Wind_Cost_Debt_2023": "Debt_Cost_2023", "Wind_Cost_Equity_2023":"Equity_Cost_2023"})
    elif technology == "Solar":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Solar_WACC_2023', 'Solar_Cost_Debt_2023', 'Solar_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Solar_WACC_2023":"WACC_2023", "Solar_Cost_Debt_2023": "Debt_Cost_2023", "Solar_Cost_Equity_2023":"Equity_Cost_2023"})
    financing_terms = pd.merge(working_dataframe, financing_values, how="left", on="index") 
    
    # Calculate the debt / concessional debt / equity shares
    financing_terms['Equity_Share'] = (100 - financing_terms['Debt_Share_2023']) / 100
    financing_terms['Concessional_Debt_Share'] = 1 / (1 + concessional_leverage)
    financing_terms['Commercial_Debt_Share'] = 1 - financing_terms['Equity_Share'] - financing_terms['Concessional_Debt_Share']
    
    # Calculate shares of final
    financing_terms['Equity_Contribution'] = financing_terms['Equity_Share'] * (financing_terms['Equity_Cost_2023'])
    financing_terms['Debt_Contribution'] = financing_terms['Commercial_Debt_Share'] * (financing_terms['Debt_Cost_2023'] ) * (1 - financing_terms['Tax_Rate']/100)
    financing_terms['Concessional_Rate'] =  (financing_terms['Required_WACC'] - financing_terms['Equity_Contribution']  - financing_terms['Debt_Contribution'])    / financing_terms['Concessional_Debt_Share'] 
    financing_terms['Concessionality'] = financing_terms['Debt_Cost_2023'] - financing_terms['Concessional_Rate']
    
    # Create a check
    financing_terms['WACC_Concessional'] =  financing_terms['Equity_Contribution'] + financing_terms['Debt_Contribution'] + financing_terms['Concessional_Rate'] * financing_terms['Concessional_Debt_Share']
    
    # Address inequalities
    financing_terms.loc[financing_terms['Concessional_Rate'] < 0, "Concessional_Rate"] = 999
    financing_terms.loc[financing_terms['Concessional_Rate'] < 0, "Concessionality"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 999, "Concessional_Rate"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 111, "Concessional_Rate"] = 111
    financing_terms.loc[financing_terms['Required_WACC'] == 999, "Concessionality"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 111, "Concessionality"] = 111
    
    # Convert back to netcdf
    financing_terms = financing_terms.set_index(["latitude", "longitude"])
    processed_data = financing_terms.to_xarray()
    
    return financing_terms, processed_data
       

def get_concessionality_v3(wacc_model, data, concessional_rate, technology):
    
    # Convert data in netcdf to pandas dataframe
    merged_dataset = xr.Dataset({
    "Required_WACC": data,
    "index": wacc_model.land_mapping
})
    working_dataframe = merged_dataset.to_dataframe().reset_index()
    
    # Import the existing costs of commercial debt, equity and the modelled debt share
    if technology == "Wind":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Onshore_Wind_WACC_2023', 'Wind_Cost_Debt_2023', 'Wind_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Onshore_Wind_WACC_2023":"WACC_2023", "Wind_Cost_Debt_2023": "Debt_Cost_2023", "Wind_Cost_Equity_2023":"Equity_Cost_2023"})
    elif technology == "Solar":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Solar_WACC_2023', 'Solar_Cost_Debt_2023', 'Solar_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Solar_WACC_2023":"WACC_2023", "Solar_Cost_Debt_2023": "Debt_Cost_2023", "Solar_Cost_Equity_2023":"Equity_Cost_2023"})
    financing_terms = pd.merge(working_dataframe, financing_values, how="left", on="index") 
    
    # Calculate debt-equity ratio
    financing_terms['Equity_Share'] = (100 - financing_terms['Debt_Share_2023']) / 100
    financing_terms['Debt_Equity_Ratio'] = financing_terms['Equity_Share'] / financing_terms['Debt_Share']
    
    # Set the cost of concessional financing
    financing_terms['Concessional_Cost_2023'] = concessional_rate
    
    
    # Calculate the share of concessional financing / commercial debt / commercial equity
    financing_terms['Concessional_Debt_Share'] = ((financing_terms['Debt_Equity_Ratio'] + 1) * financing_terms['Required_WACC'] - financing_terms['Equity_Cost_2023'] - financing_terms['Debt_Cost_2023']* (1 - financing_terms['Tax_Rate']/100)) / ((financing_terms['Debt_Equity_Ratio'] + 1) * financing_terms['Concessional_Cost_2023'] - financing_terms['Equity_Cost_2023'] - financing_terms['Debt_Cost_2023']* (1 - financing_terms['Tax_Rate']/100))
    financing_terms['Commercial_Debt_Share'] = (1 - financing_terms['Concessional_Debt_Share']) / (financing_terms['Debt_Equity_Ratio'] + 1)
    financing_terms['Commercial_Equity_Share'] = 1 - financing_terms['Concessional_Debt_Share'] - financing_terms['Commercial_Debt_Share']
    
    
    
    # Calculate the debt / concessional debt / equity shares
    financing_terms['Equity_Share'] = (100 - financing_terms['Debt_Share_2023']) / 100
    financing_terms['Concessional_Debt_Share'] = 1 / (1 + concessional_leverage)
    financing_terms['Commercial_Debt_Share'] = 1 - financing_terms['Equity_Share'] - financing_terms['Concessional_Debt_Share']
    
    # Calculate shares of final
    financing_terms['Equity_Contribution'] = financing_terms['Commercial_Equity_Share'] * (financing_terms['Equity_Cost_2023'])
    financing_terms['Debt_Contribution'] = financing_terms['Commercial_Debt_Share'] * (financing_terms['Debt_Cost_2023'] ) * (1 - financing_terms['Tax_Rate']/100)
    financing_terms['Concessional_Contribution'] = financing_terms['Concessional_Debt_Share'] * financing_terms['Concessional_Cost_2023']
    financing_terms['Concessionality'] = financing_terms['Debt_Cost_2023'] - financing_terms['Concessional_Cost_2023']
    
    # Create a check
    financing_terms['WACC_Concessional'] =  financing_terms['Equity_Contribution'] + financing_terms['Debt_Contribution'] + financing_terms['Concessional_Contribution']
    
    # Address inequalities
    financing_terms.loc[financing_terms['Required_WACC'] == 999, "Concessional_Rate"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 111, "Concessional_Rate"] = 111
    financing_terms.loc[financing_terms['Required_WACC'] == 999, "Concessionality"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 111, "Concessionality"] = 111
    
    # Convert back to netcdf
    financing_terms = financing_terms.set_index(["latitude", "longitude"])
    processed_data = financing_terms.to_xarray()
    
    return financing_terms, processed_data    
    
    
    
wacc_concessional_solar = xr.where(solar_benchmark_results['Calculated_LCOE'] < solar_benchmark_results['Benchmark_LCOE'], 111, solar_benchmark_wacc)
concessional_calcs, processed_data_solar = get_concessionality_v2(wacc_model, wacc_concessional_solar, 1.8, "Solar")
wacc_model.plot_data_shading(processed_data_solar['Concessionality'], processed_data_solar.latitude, processed_data_solar.longitude, special_value = 999, hatch_label = "Unable to reach benchmark\nof US$68/MWh - system costs\nusing concessional financing", special_value_2 = 111, hatch_label_2="Below US$102.5/MWh\nwithout concessional financing", tick_values = [0, 5, 10, 15, 20], cmap="YlOrRd", title="Required\nConcessional\nRate Below\nMarket (%)\n", filename="CONC_FINANCE_SOLAR", graphmarking="a", extend_set="max")
print(np.nanmean(xr.where((processed_data_solar['Concessional_Rate'] == 111) | (processed_data_solar['Concessional_Rate'] ==999), np.nan, processed_data_solar['Concessional_Rate'])))
print(np.nanmean(xr.where((processed_data_solar['Concessionality'] == 111) | (processed_data_solar['Concessionality'] ==999), np.nan, processed_data_solar['Concessionality'])))

wacc_concessional_wind = xr.where(wind_benchmark_results['Calculated_LCOE'] < wind_benchmark_results['Benchmark_LCOE'], 111, wind_benchmark_wacc)
concessional_calcs, processed_data_wind = get_concessionality_v2(wacc_model, wacc_concessional_wind, 1.8, "Wind")
wacc_model.plot_data_shading(processed_data_wind['Concessionality'].drop_sel(latitude=0), processed_data_wind.drop_sel(latitude=0).latitude, processed_data_wind.longitude, special_value = 999, hatch_label = "Unable to reach benchmark\nof US$102.5/MWh - system costs\nusing concessional financing", special_value_2 = 111, hatch_label_2="Below US$102.5/MWh - system\ncosts without concessional financing", tick_values = [0, 5, 10, 15, 20], cmap="YlGnBu", title="Required\nConcessional\nRate Below\nMarket (%)\n", filename="CONC_FINANCE_WIND", graphmarking="b", extend_set="max")
print(np.nanmean(xr.where((processed_data_wind['Concessional_Rate'] == 111) | (processed_data_wind['Concessional_Rate'] ==999), np.nan, processed_data_wind['Concessional_Rate'])))
print(np.nanmean(xr.where((processed_data_wind['Concessionality'] == 111) | (processed_data_wind['Concessionality'] ==999), np.nan, processed_data_wind['Concessionality'])))






def get_concessionality_v3(wacc_model, data, concessional_rate, technology):
    
    # Convert data in netcdf to pandas dataframe
    merged_dataset = xr.Dataset({
    "Required_WACC": data,
    "index": wacc_model.land_mapping
})
    working_dataframe = merged_dataset.to_dataframe().reset_index()
    
    # Import the existing costs of commercial debt, equity and the modelled debt share
    if technology == "Wind":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Onshore_Wind_WACC_2023', 'Wind_Cost_Debt_2023', 'Wind_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Onshore_Wind_WACC_2023":"WACC_2023", "Wind_Cost_Debt_2023": "Debt_Cost_2023", "Wind_Cost_Equity_2023":"Equity_Cost_2023"})
    elif technology == "Solar":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Solar_WACC_2023', 'Solar_Cost_Debt_2023', 'Solar_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Solar_WACC_2023":"WACC_2023", "Solar_Cost_Debt_2023": "Debt_Cost_2023", "Solar_Cost_Equity_2023":"Equity_Cost_2023"})
    financing_terms = pd.merge(working_dataframe, financing_values, how="left", on="index") 
    
    # Calculate debt-equity ratio
    financing_terms['Equity_Share_2023'] = (100 - financing_terms['Debt_Share_2023']) 
    financing_terms['Debt_Equity_Ratio'] =  financing_terms['Debt_Share_2023'] / financing_terms['Equity_Share_2023']
    
    # Set the cost of concessional financing
    financing_terms['Concessional_Cost_2023'] = concessional_rate
    
    
    # Calculate the share of concessional financing / commercial debt / commercial equity
    financing_terms['Concessional_Debt_Share'] = ((financing_terms['Debt_Equity_Ratio'] + 1) * financing_terms['Required_WACC'] - (financing_terms['Equity_Cost_2023'] / financing_terms['Debt_Equity_Ratio']) - financing_terms['Debt_Cost_2023']) / ((financing_terms['Debt_Equity_Ratio'] + 1) * financing_terms['Concessional_Cost_2023'] - (financing_terms['Equity_Cost_2023'] / financing_terms['Debt_Equity_Ratio']) - financing_terms['Debt_Cost_2023'])
    financing_terms['Commercial_Debt_Share'] = (1 - financing_terms['Concessional_Debt_Share']) / (financing_terms['Debt_Equity_Ratio'] + 1) 
    financing_terms['Commercial_Equity_Share'] = 1 - financing_terms['Concessional_Debt_Share'] - financing_terms['Commercial_Debt_Share'] 
    
    # Calculate shares of final
    financing_terms['Equity_Contribution'] = financing_terms['Commercial_Equity_Share'] * (financing_terms['Equity_Cost_2023'])
    financing_terms['Debt_Contribution'] = financing_terms['Commercial_Debt_Share'] * (financing_terms['Debt_Cost_2023'] ) 
    financing_terms['Concessional_Contribution'] = financing_terms['Concessional_Debt_Share'] * financing_terms['Concessional_Cost_2023']
    financing_terms['Concessionality'] = financing_terms['Debt_Cost_2023'] - financing_terms['Concessional_Cost_2023']
    
    # Create a check
    financing_terms['WACC_Concessional'] =  financing_terms['Equity_Contribution'] + financing_terms['Debt_Contribution'] + financing_terms['Concessional_Contribution']
    
    # Address inequalities
    financing_terms['Concessional_Debt_Share'] = financing_terms['Concessional_Debt_Share'] * 100
    financing_terms.loc[financing_terms['Required_WACC'] == 999, "Concessional_Debt_Share"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 111, "Concessional_Debt_Share"] = 111
    
    
    # Convert back to netcdf
    financing_terms = financing_terms.set_index(["latitude", "longitude"])
    processed_data = financing_terms.to_xarray()
    
    return financing_terms, processed_data    
    
    
    
wacc_concessional_solar = xr.where(solar_system_results['Calculated_LCOE'] < solar_system_results['Benchmark_LCOE'], 111, solar_system_wacc)
concessional_calcs, processed_data_solar = get_concessionality_v3(wacc_model, wacc_concessional_solar, 1, "Solar")
wacc_model.plot_data_shading(processed_data_solar['Concessional_Debt_Share'], processed_data_solar.latitude, processed_data_solar.longitude, special_value = 999, hatch_label = "Unable to reach benchmark\nof US$68/MWh - system costs\nusing concessional financing", special_value_2 = 111, hatch_label_2="Below US$68/MWh - system\ncosts without concessional financing", tick_values = [0, 25, 50, 75, 100], cmap="YlOrRd", title="Required\nShare Of\nConcessional\nFinancing (%)\n", filename="CONC_FINANCE_SOLAR", graphmarking="a", extend_set="neither")
print(np.nanmean(xr.where((processed_data_solar['Concessional_Debt_Share'] == 111) | (processed_data_solar['Concessional_Debt_Share'] ==999), np.nan, processed_data_solar['Concessional_Debt_Share'])))
print(np.nanmean(xr.where((processed_data_solar['Concessional_Debt_Share'] == 111) | (processed_data_solar['Concessional_Debt_Share'] ==999), np.nan, processed_data_solar['Concessional_Debt_Share'])))

wacc_concessional_wind = xr.where(wind_system_results['Calculated_LCOE'] < wind_system_results['Benchmark_LCOE'], 111, wind_system_wacc)
concessional_calcs_v1, processed_data_wind = get_concessionality_v3(wacc_model, wacc_concessional_wind, 1, "Wind")
wacc_model.plot_data_shading(processed_data_wind['Concessional_Debt_Share'].drop_sel(latitude=0), processed_data_wind.drop_sel(latitude=0).latitude, processed_data_wind.longitude, special_value = 999, hatch_label = "Unable to reach benchmark\nof US$68/MWh - system costs\nusing concessional financing", special_value_2 = 111, hatch_label_2="Below US$68/MWh - system\ncosts without concessional financing", tick_values = [0, 25, 50, 75, 100], cmap="YlGnBu", title="Required\nShare Of\nConcessional\nFinancing (%)\n", filename="CONC_FINANCE_WIND", graphmarking="b", extend_set="neither")
print(np.nanmean(xr.where((processed_data_wind['Concessional_Debt_Share'] == 111) | (processed_data_wind['Concessional_Debt_Share'] ==999), np.nan, processed_data_wind['Concessional_Debt_Share'])))
print(np.nanmean(xr.where((processed_data_wind['Concessional_Debt_Share'] == 111) | (processed_data_wind['Concessional_Debt_Share'] ==999), np.nan, processed_data_wind['Concessional_Debt_Share'])))




## GLOBAL COSTS WITH SYSTEM COSTS INCLUDED
    
# Estimate the required WACC to reach cost parity with global costs
solar_system_results = wacc_model.calculate_required_waccs(solar_results, benchmark_lcoe = 68, technology="Solar")
wind_system_results = wacc_model.calculate_required_waccs(wind_results, benchmark_lcoe = 68, technology="Wind")

# Calculate WACC 
solar_system_wacc = solar_system_results['Benchmark_WACC']
wind_system_wacc = wind_system_results['Benchmark_WACC']

# Compare WACC to the reductions required
wacc_system_solar = xr.where(np.isnan(solar_system_results['Calculated_LCOE'])==True, np.nan, xr.where(solar_system_results['Estimated_WACC'] > solar_system_wacc,solar_system_results['Estimated_WACC'] - solar_system_wacc, 999))
wacc_system_wind = xr.where(np.isnan(wind_system_results['Calculated_LCOE'])==True, np.nan,xr.where(wind_system_results['Estimated_WACC'] > wind_system_wacc, wind_system_results['Estimated_WACC'] - wind_system_wacc, 999))

# Apply hatching to where the cost is unachievable
wacc_reductions_solar_system = xr.where(solar_system_results['Benchmark_WACC'] == 999, 999, wacc_system_solar)
wacc_reductions_wind_system = xr.where(wind_system_results['Benchmark_WACC'] == 999, 999, wacc_system_wind)

# Apply hatching to locations already below the system cost of renewables
wacc_reductions_solar_system = xr.where(solar_system_results['Calculated_LCOE'] < solar_system_results['Benchmark_LCOE'], 111, wacc_reductions_solar_system)
wacc_reductions_wind_system = xr.where(wind_system_results['Calculated_LCOE'] < wind_system_results['Benchmark_LCOE'], 111, wacc_reductions_wind_system)

# Remove latitude of zero from wind
wacc_reductions_wind_system= wacc_reductions_wind_system.drop_sel(latitude=0)

plot_wacc_sys_results = input("Start Plotting LCOE (No System Costs)? Yes/No")
if plot_wacc_sys_results == "Yes":

    # Plot for when system costs are included
    wacc_model.plot_data_shading(wacc_reductions_wind_system, wacc_reductions_wind.latitude, wacc_reductions_wind.longitude, tick_values = [0, 5, 10, 15, 20, 25], special_value=999, hatch_label="Above US$68/MWh\nwith system costs\nat a WACC of 0%", special_value_2 = 111, hatch_label_2 = "Below US$68/MWh with\nsystem costs included", title="Reductions\nin WACC\n Required (%)\n", cmap="YlGnBu", graphmarking="b", extend_set="neither", filename = output_folder + "Wind_SYS_Cost_High") 
    wacc_model.plot_data_shading(wacc_reductions_solar_system, wacc_reductions_solar.latitude, wacc_reductions_solar.longitude, tick_values = [0, 5, 10, 15, 20, 25], special_value=999, hatch_label="Above US$68/MWh\nwith system costs\nat a WACC of 0%", special_value_2 = 111, hatch_label_2 = "Below US$68/MWh with\nsystem costs included", title="Reductions\nin WACC\n Required (%)\n", cmap="YlOrRd", graphmarking="a", extend_set="neither", filename = output_folder + "Solar_SYS_Cost_High")





def get_concessionality_v3(wacc_model, data, concessional_rate, technology):
    
    # Convert data in netcdf to pandas dataframe
    merged_dataset = xr.Dataset({
    "Required_WACC": data,
    "index": wacc_model.land_mapping
})
    working_dataframe = merged_dataset.to_dataframe().reset_index()
    
    # Import the existing costs of commercial debt, equity and the modelled debt share
    if technology == "Wind":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Onshore_Wind_WACC_2023', 'Wind_Cost_Debt_2023', 'Wind_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Onshore_Wind_WACC_2023":"WACC_2023", "Wind_Cost_Debt_2023": "Debt_Cost_2023", "Wind_Cost_Equity_2023":"Equity_Cost_2023"})
    elif technology == "Solar":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Solar_WACC_2023', 'Solar_Cost_Debt_2023', 'Solar_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Solar_WACC_2023":"WACC_2023", "Solar_Cost_Debt_2023": "Debt_Cost_2023", "Solar_Cost_Equity_2023":"Equity_Cost_2023"})
    financing_terms = pd.merge(working_dataframe, financing_values, how="left", on="index") 
    
    # Calculate debt-equity ratio
    financing_terms['Equity_Share_2023'] = (100 - financing_terms['Debt_Share_2023']) 
    financing_terms['Debt_Equity_Ratio'] =  financing_terms['Debt_Share_2023'] / financing_terms['Equity_Share_2023']
    
    # Set the cost of concessional financing 
    financing_terms['Concessional_Cost_2023'] = concessional_rate 
    
    
    # Calculate the share of concessional financing / commercial debt / commercial equity
    numerator =  (financing_terms['Debt_Equity_Ratio'] + 1) * financing_terms['Required_WACC'] - financing_terms['Debt_Cost_2023'] * financing_terms['Debt_Equity_Ratio'] * (1 - financing_terms['Tax_Rate']/100) - financing_terms['Equity_Cost_2023']
    denominator = ((financing_terms['Debt_Equity_Ratio'] + 1) * financing_terms['Concessional_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100) - financing_terms['Equity_Cost_2023'] - financing_terms['Debt_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100) * financing_terms['Debt_Equity_Ratio'])
    financing_terms['Concessional_Debt_Share'] = numerator / denominator
    financing_terms['Commercial_Equity_Share'] = (1 - financing_terms['Concessional_Debt_Share']) / (financing_terms['Debt_Equity_Ratio'] + 1) 
    financing_terms['Commercial_Debt_Share'] = 1 - financing_terms['Concessional_Debt_Share'] - financing_terms['Commercial_Equity_Share'] 
    
    # Calculate shares of final
    financing_terms['Equity_Contribution'] = financing_terms['Commercial_Equity_Share'] * (financing_terms['Equity_Cost_2023'])
    financing_terms['Debt_Contribution'] = financing_terms['Commercial_Debt_Share'] * (financing_terms['Debt_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100) ) 
    financing_terms['Concessional_Contribution'] = financing_terms['Concessional_Debt_Share'] * financing_terms['Concessional_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100)
    financing_terms['Concessionality'] = financing_terms['Debt_Cost_2023'] - financing_terms['Concessional_Cost_2023']
    
    # Create a check
    financing_terms['Total_Check'] =  financing_terms['Concessional_Debt_Share'] + financing_terms['Commercial_Debt_Share'] + financing_terms['Commercial_Equity_Share']
    financing_terms['WACC_Concessional'] =  financing_terms['Equity_Contribution'] + financing_terms['Debt_Contribution'] + financing_terms['Concessional_Contribution']
    
    # Address inequalities
    financing_terms['Concessional_Debt_Share'] = financing_terms['Concessional_Debt_Share'] * 100
    financing_terms.loc[financing_terms['Required_WACC'] == 999, "Concessional_Debt_Share"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 111, "Concessional_Debt_Share"] = 111
    financing_terms.loc[financing_terms['Required_WACC'] < concessional_rate, "Concessional_Debt_Share"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] < concessional_rate, "Commercial_Equity_Share"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] < concessional_rate, "Commercial_Debt_Share"] = 999
    
    # Convert back to netcdf
    financing_terms = financing_terms.set_index(["latitude", "longitude"])
    processed_data = financing_terms.to_xarray()
    
    return financing_terms, processed_data    
    
    
    
wacc_concessional_solar = xr.where(solar_system_results['Calculated_LCOE'] < solar_system_results['Benchmark_LCOE'], 111, solar_system_wacc)
concessional_calcs, processed_data_solar = get_concessionality_v3(wacc_model, wacc_concessional_solar, 1, "Solar")
wacc_model.plot_data_shading(processed_data_solar['Concessional_Debt_Share'], processed_data_solar.latitude, processed_data_solar.longitude, special_value = 999, hatch_label = "Above US$68/MWh -\nsystem costs with 100%\nconcessional financing", special_value_2 = 111, hatch_label_2="Below US$68/MWh -\nsystem costs without\nconcessional financing", tick_values = [0, 25, 50, 75, 100], cmap="YlOrRd", title="Required\nShare Of\nConcessional\nFinancing (%)\n", filename="CONC_FINANCE_SOLAR_SYS", graphmarking="a", extend_set="neither")
print(np.nanmean(xr.where((processed_data_solar['Concessional_Debt_Share'] == 111) | (processed_data_solar['Concessional_Debt_Share'] ==999), np.nan, processed_data_solar['Concessional_Debt_Share'])))
print(np.nanmean(xr.where((processed_data_solar['Concessional_Debt_Share'] == 111) | (processed_data_solar['Concessional_Debt_Share'] ==999), np.nan, processed_data_solar['Concessional_Debt_Share'])))


wacc_concessional_wind = xr.where(wind_system_results['Calculated_LCOE'] < wind_system_results['Benchmark_LCOE'], 111, wind_system_wacc)
concessional_calcs_v1, processed_data_wind = get_concessionality_v3(wacc_model, wacc_concessional_wind, 1, "Wind")
wacc_model.plot_data_shading(processed_data_wind['Concessional_Debt_Share'].drop_sel(latitude=0), processed_data_wind.drop_sel(latitude=0).latitude, processed_data_wind.longitude, special_value = 999, hatch_label = "Above US$68/MWh -\nsystem costs with 100%\nconcessional financing", special_value_2 = 111, hatch_label_2="Below US$68/MWh -\nsystem costs without\nconcessional financing", tick_values = [0, 25, 50, 75, 100], cmap="YlGnBu", title="Required\nShare Of\nConcessional\nFinancing (%)\n", filename="CONC_FINANCE_WIND_SYS", graphmarking="b", extend_set="neither")
print(np.nanmean(xr.where((processed_data_wind['Concessional_Debt_Share'] == 111) | (processed_data_wind['Concessional_Debt_Share'] ==999), np.nan, processed_data_wind['Concessional_Debt_Share'])))
print(np.nanmean(xr.where((processed_data_wind['Concessional_Debt_Share'] == 111) | (processed_data_wind['Concessional_Debt_Share'] ==999), np.nan, processed_data_wind['Concessional_Debt_Share'])))


wacc_concessional_solar = xr.where(solar_benchmark_results['Calculated_LCOE'] < solar_benchmark_results['Benchmark_LCOE'], 111, solar_benchmark_wacc)
concessional_calcs, processed_data_solar_benchmark = get_concessionality_v3(wacc_model, wacc_concessional_solar, 1, "Solar")
wacc_model.plot_data_shading(processed_data_solar_benchmark['Concessional_Debt_Share'], processed_data_solar_benchmark.latitude, processed_data_solar_benchmark.longitude, special_value = 999, hatch_label = "Above US$68/MWh\nwith 100% concessional \nfinancing", special_value_2 = 111, hatch_label_2="Below US$68/MWh\nwithout concessional \nfinancing", tick_values = [0, 25, 50, 75, 100], cmap="YlOrRd", title="Required\nShare Of\nConcessional\nFinancing (%)\n", filename="CONC_FINANCE_SOLAR", graphmarking="a", extend_set="neither")
print(np.nanmean(xr.where((processed_data_solar_benchmark['Concessional_Debt_Share'] == 111) | (processed_data_solar_benchmark['Concessional_Debt_Share'] ==999), np.nan, processed_data_solar_benchmark['Concessional_Debt_Share'])))
print(np.nanmean(xr.where((processed_data_solar['Concessional_Debt_Share'] == 111) | (processed_data_solar_benchmark['Concessional_Debt_Share'] ==999), np.nan, processed_data_solar_benchmark['Concessional_Debt_Share'])))


wacc_concessional_wind = xr.where(wind_benchmark_results['Calculated_LCOE'] < wind_benchmark_results['Benchmark_LCOE'], 111, wind_benchmark_wacc)
concessional_calcs_v1, processed_data_wind_benchmark = get_concessionality_v3(wacc_model, wacc_concessional_wind, 1, "Wind")
wacc_model.plot_data_shading(processed_data_wind_benchmark['Concessional_Debt_Share'].drop_sel(latitude=0), processed_data_wind_benchmark.drop_sel(latitude=0).latitude, processed_data_wind_benchmark.longitude, special_value = 999, hatch_label = "Above US$68/MWh\nwith 100% concessional \nfinancing", special_value_2 = 111, hatch_label_2="Below US$68/MWh\nwithout concessional \nfinancing", tick_values = [0, 25, 50, 75, 100], cmap="YlGnBu", title="Required\nShare Of\nConcessional\nFinancing (%)\n", filename="CONC_FINANCE_WIND", graphmarking="b", extend_set="neither")
print(np.nanmean(xr.where((processed_data_wind_benchmark['Concessional_Debt_Share'] == 111) | (processed_data_wind_benchmark['Concessional_Debt_Share'] ==999), np.nan, processed_data_wind_benchmark['Concessional_Debt_Share'])))
print(np.nanmean(xr.where((processed_data_wind_benchmark['Concessional_Debt_Share'] == 111) | (processed_data_wind_benchmark['Concessional_Debt_Share'] ==999), np.nan, processed_data_wind_benchmark['Concessional_Debt_Share'])))








def produce_csv_data(data, required_variables, name):
    
    extracted_data = data.merge(wacc_model.land_mapping).rename({"land":"Country", 'electricity_production':"Average_Electricity_Production", "latitude":"Latitude", "longitude":"Longitude"})
    variables = ["Country"]
    variables.append(required_variables)
    extracted_data[variables].to_dataframe().dropna().to_csv("./SUPPLEMENTARY/" + name)
    
produce_csv_data(solar_results, "Calculated_LCOE", "Figure_4a.csv")
produce_csv_data(wind_results, "Calculated_LCOE", "Figure_4b.csv")
produce_csv_data(solar_results, "Uniform_LCOE", "Figure_4c.csv")
produce_csv_data(wind_results, "Uniform_LCOE", "Figure_4d.csv")

def produce_csv_data_v2(data, required_variables, name):
    
    extracted_data = data.merge(wacc_model.land_mapping).rename({"land":"Country", "latitude":"Latitude", "longitude":"Longitude"})
    variables = ["Country"]
    variables.append(required_variables)
    extracted_data[variables].to_dataframe().dropna().to_csv("./SUPPLEMENTARY/" + name)

produce_csv_data_v2(wacc_reductions_solar_system.to_dataset(name="WACC_Reductions"), "WACC_Reductions", "Figure_5a.csv")
produce_csv_data_v2(wacc_reductions_wind_system.to_dataset(name="WACC_Reductions"), "WACC_Reductions", "Figure_5b.csv")
produce_csv_data_v2(processed_data_solar[['Concessional_Debt_Share']], "Concessional_Debt_Share", "Figure_6a.csv")
produce_csv_data_v2(processed_data_wind[['Concessional_Debt_Share']], "Concessional_Debt_Share", "Figure_6b.csv")


produce_csv_data_v2(wacc_reductions_solar.to_dataset(name="WACC_Reductions"), "WACC_Reductions", "Figure_ED2a.csv")
produce_csv_data_v2(wacc_reductions_wind.to_dataset(name="WACC_Reductions"), "WACC_Reductions", "Figure_ED2b.csv")
produce_csv_data_v2(processed_data_solar_benchmark[['Concessional_Debt_Share']], "Concessional_Debt_Share", "Figure_ED3a.csv")
produce_csv_data_v2(processed_data_wind_benchmark[['Concessional_Debt_Share']], "Concessional_Debt_Share", "Figure_ED3b.csv")








from sklearn import linear_model
import scipy.stats as stats

def run_regression_model(CoCModel, data, future_data, selected_parameter, influencing_parameters, future_parameters, year, rf_change, technology, print_values=None, filename=None, plot=None):
    
    
    # Filter training data
    filtered_data = data.dropna(subset=influencing_parameters)
    filtered_data = filtered_data.dropna(subset=[selected_parameter])

    # Filter future data
    filtered_future_data = future_data.dropna(subset=future_parameters)

    # Training Data
    TD_inputs = filtered_data[influencing_parameters].to_numpy()
    TD_values = filtered_data[selected_parameter].to_numpy()
    
    # New inputs
    NEW_inputs = filtered_future_data[future_parameters].to_numpy()
    
    # Set up linear regression model
    regr = linear_model.Ridge(alpha=100)
    regr.fit(TD_inputs, TD_values)

    # Extract coefficients 
    coefficients = regr.coef_
    coefficients_list = list(coefficients)
    intercept = regr.intercept_
    r_squared = regr.score(TD_inputs, TD_values)

    # Print values, if desired
    if print_values is not None:
        print("Intercept:", intercept)
        print("Coefficients:", coefficients)
        print("R-squared:", r_squared)


    # Calculate residuals
    estimated_values = regr.predict(TD_inputs)
    residuals = TD_values - estimated_values

    # Predict future values and account for change in risk free rate
    predicted_values = regr.predict(NEW_inputs)
    predicted_values = predicted_values + rf_change

    # Calculate standard error
    n = len(TD_inputs)
    p = TD_inputs.shape[1]  # Number of predictors (features)
    standard_error = np.sqrt(np.sum(residuals ** 2) / (n - p - 2))

    # Calculate t-values
    confidence = 0.90
    t_value = stats.t.ppf((1 + confidence) / 2, n - p - 1)

    # Step 6: Compute the prediction interval
    # Calculate X
    X = np.column_stack((np.ones(TD_inputs.shape[0]),TD_inputs))
    X_t = np.transpose(X)
    
    # Calculate X_new
    X_new = np.column_stack((np.ones(NEW_inputs.shape[0]),NEW_inputs))
    X_new_T = np.transpose(X_new)
    
    # Calculate intermediate steps 
    XtX = np.matmul(X_t, X)
    XtX_inv = np.linalg.inv(XtX)
    intermediate = np.matmul(X_new, XtX_inv)
    matrix_calculation = np.einsum('ij,ij->i', intermediate, X_new)
    interval = t_value * standard_error * np.sqrt(1 + matrix_calculation)
    interval_confidence = t_value * standard_error * np.sqrt(1 + matrix_calculation)


    # Calculate prediction intervals
    lower_bound = predicted_values - interval
    upper_bound = predicted_values + interval

    # Merge all the data
    filtered_future_data[technology +'_Common_Risk_'+year] = pd.Series(predicted_values, index=filtered_future_data.index)
    filtered_future_data[technology+'_Common_Risk_LB_'+year] = pd.Series(lower_bound, index=filtered_future_data.index)
    filtered_future_data[technology+'_Common_Risk_UB_'+year] = pd.Series(upper_bound, index=filtered_future_data.index)
    filtered_future_data[technology+'_Confidence_Interval_'+year] = pd.Series(interval_confidence, index=filtered_future_data.index)
    
    # Plotting 
    if plot is not None:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.scatter(filtered_data[selected_parameter], estimated_values, color='blue', label='Predicted')
        ax.plot(filtered_data[selected_parameter].values, filtered_data[selected_parameter].values, color='red', linestyle='--', label='Measure of Accuracy')
        if variable_name is None:
            variable_name = ""
        plt.xlabel('Actual ' + variable_name, fontsize=20)
        plt.ylabel('Predicted ' + variable_name, fontsize=20)
        plt.title('Actual vs Predicted ' + variable_name, fontsize=20)
        plt.text(0.5, 0.98, f"R2 = {r_squared:0.2f}", fontsize=20, ha='center', va='top', transform=ax.transAxes) 
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

    regressed_data = pd.merge(filtered_future_data, data, on="Country code", how="left")
    regressed_data.drop([col for col in regressed_data.columns if '_y' in col],axis=1,inplace=True)
    regressed_data.drop([col for col in regressed_data.columns if '_x' in col],axis=1,inplace=True)

    return regressed_data


def estimate_wacc(CoCModel, data, year, technology, erp=None):
    
    
    # Get Debt and Equity Risk Premiums for the year
    if int(year) > 2023:
        equity_risk = erp
    else:
        crp_data = CoCModel.pull_CRP_data(str(year))
        erp = crp_data[crp_data["Country code"] == "ERP"]["CRP_"+year].values[0]
        
    # Calculate Debt Share
    data = CoCModel.get_debt_share(data, year)
    data['Debt_Share_'+year] = data['Debt_Share_'+year]/100
        
    # Calculate Cost of Debt
    data[technology + "_Cost_Debt_" + year] = data[technology + "_Common_Risk_" + year] + CoCModel.debt_margin
    data[technology + "_Cost_Debt_LB_" + year] = data[technology + "_Common_Risk_LB_" + year] + CoCModel.debt_margin
    data[technology + "_Cost_Debt_UB_" + year] = data[technology + "_Common_Risk_UB_" + year] + CoCModel.debt_margin
    
    # Calculate Cost of Debt
    data[technology + "_Cost_Equity_" + year] = data[technology + "_Common_Risk_" + year] + erp
    data[technology + "_Cost_Equity_LB_" + year] = data[technology + "_Common_Risk_LB_" + year] + erp
    data[technology + "_Cost_Equity_UB_" + year] = data[technology + "_Common_Risk_UB_" + year] + erp
    
    # Calculate WACC
    data[technology + "_WACC_" + year] = data[technology + "_Cost_Debt_" + year] * data["Debt_Share_" + year] * (1 - data["Tax_Rate"]/100) + (1 - data["Debt_Share_" + year]) * data[technology + "_Cost_Equity_" + year]
    data[technology + "_WACC_LB" + year] = data[technology + "_Cost_Debt_LB_" + year] * data["Debt_Share_" + year] * (1 - data["Tax_Rate"]/100) + (1 - data["Debt_Share_" + year]) * data[technology + "_Cost_Equity_LB_" + year]
    data[technology + "_WACC_UB" + year] = data[technology + "_Cost_Debt_UB_" + year] * data["Debt_Share_" + year] * (1 - data["Tax_Rate"]/100) + (1 - data["Debt_Share_" + year]) * data[technology + "_Cost_Equity_UB_" + year]
    
    return data

def collate_future_data(CoCModel, year):
    
    # Get CRPs
    if int(year) > 2023:
        crp_data = CoCModel.pull_CRP_data("2023")
        crp_data.rename(columns={"CRP_2023":"CRP_" + year})
    else: 
        crp_data = CoCModel.pull_CRP_data(str(year))
    
    # Get Renewable Penetration
    def fill_missing_RE_values(data, previous_year, year):

        # Set Country Code as index
        data.set_index('Country code', inplace=True)
        previous_year.set_index('Country code', inplace=True)

        # Fill missing values for 2023 with 2022 data
        data = pd.merge(data, previous_year, on="Country code", how="outer")
        data['Penetration_2023'] = data['Penetration_2023'].fillna(data['Penetration_'+year])

        # Reset index if needed
        data.reset_index(inplace=True)

        return data
    
    def drop_y(df):
        # list comprehension of the cols that end with '_y'
        to_drop = [x for x in df if x.endswith('_y')]
        df.drop(to_drop, axis=1, inplace=True)
    
     # Get generation data for the specified year
    re_data_solar = CoCModel.pull_generation_data_v2(year, "Solar")
    re_data_wind = CoCModel.pull_generation_data_v2(year, "Wind")

    # Fill missing datapoints with 2022 / 2023 data. 
    re_solar_2022 = CoCModel.pull_generation_data_v2("2022", "Solar")
    re_wind_2022 = CoCModel.pull_generation_data_v2("2022", "Wind")
    re_data_solar = fill_missing_RE_values(re_data_solar, re_solar_2022, "2022")
    re_data_wind = fill_missing_RE_values(re_data_wind, re_wind_2022, "2022")
    re_solar_2021 = CoCModel.pull_generation_data_v2("2021", "Solar")
    re_wind_2021= CoCModel.pull_generation_data_v2("2021", "Wind")
    re_data_solar = fill_missing_RE_values(re_data_solar, re_solar_2021, "2021")
    re_data_wind = fill_missing_RE_values(re_data_wind, re_wind_2021, "2021")
    
     # Check if it is a projection or not
    if int(year) > 2023:
        
        # Calculate number of years
        num_years = int(year)-2022
        
        # Calculate increase in capacity
        re_data_solar[["Capacity_"+year,"Penetration_"+year]] = re_data_solar[["Capacity_2022","Penetration_2022"]] * (1+ percentage_increase) ^ num_years
        re_data_wind[["Capacity_"+year,"Penetration_"+year]] = re_data_wind[["Capacity_2022","Penetration_2022"]] * (1+ percentage_increase) ^ num_years
        
    # Add Prefixes
    re_data_solar[["Solar_Capacity_"+year, "Solar_Penetration_" + year]] =  re_data_solar[["Capacity_"+year, "Penetration_" + year]]
    re_data_wind[["Wind_Capacity_"+year, "Wind_Penetration_" + year]] =  re_data_wind[["Capacity_"+year, "Penetration_" + year]]
    
    # Collate the renewables data
    collated_renewables_data = pd.merge(re_data_solar[["Country code", "Solar_Capacity_"+year, "Solar_Penetration_"+year]], re_data_wind[["Country code", "Wind_Capacity_"+year, "Wind_Penetration_"+year]], on="Country code", how="outer")
    
    
    # Get RISE
    rise_data = CoCModel.extract_rise_data(int(year))
    rise_data = rise_data.rename(columns={c:c+"_" + year for c in rise_data.columns if "RISE" in c})
    
    # Collate data
    collated_data = pd.merge(crp_data, collated_renewables_data, how="left", on="Country code")
    collated_data = pd.merge(collated_data, rise_data, how="left", on="Country code", suffixes=('', '_y'))
    drop_y(collated_data)
    future_data = collated_data                  
    
    
    return future_data

def estimate_future_waccs_v4(CoCModel, selected_parameter_wind, selected_parameter_solar, influencing_parameters_wind, influencing_parameters_solar, year, interest_rate=None, percentage_increase=None):


    # Get 2021 data from calling separate function
    irena_data_extracted = CoCModel.estimate_irena_premiums()
    irena_data_extracted = irena_data_extracted.rename(columns={c:c+"_2021" for c in irena_data_extracted.columns if "RISE" in c})
    irena_data_extracted = irena_data_extracted.rename(columns={"Wind_Common_Risk_2021": "Onshore_Wind_Common_Risk_2021", "Wind_Cost_Equity_2021":"Onshore_Wind_Cost_Equity_2021", "Wind_Cost_Debt_2021":"Onshore_Wind_Cost_Debt_2021"})
    
    # Calculate 2021 risk-free rate
    US_IR = CoCModel.OECD_IR[CoCModel.OECD_IR['Country code'] == "USA"]
    rf_rate_2021 =  US_IR['2021'].values[0]
    
    # Get desired year of data
    future_data = collate_future_data(CoCModel, year)
    
    # Calculate desired year of data risk free rate
    if int(year) > 2023:
        rf_rate = interest_rate
    else:
        rf_rate =  US_IR[year].values[0]
    
    # Risk free rate change
    rf_change = rf_rate - rf_rate_2021

    # Call the regression function with the corresponding parameters
    solar_estimates = run_regression_model(CoCModel, irena_data_extracted, future_data, selected_parameter_solar, influencing_parameters_solar, ["Solar_Penetration_"+year, "CRP_" + year, "RISE_OVERALL_"+year], year, rf_change, "Solar", print_values="True", filename=None, plot=None)
    wind_estimates = run_regression_model(CoCModel, irena_data_extracted, future_data, selected_parameter_wind, influencing_parameters_wind, ["Wind_Penetration_"+year, "CRP_" + year, "RISE_OVERALL_"+year], year, rf_change, "Onshore_Wind", print_values="True", filename=None, plot=None)
    
    # Call function to estimate debt share, cost of debt, cost of equity and WACC
    solar_estimated_data = estimate_wacc(CoCModel, solar_estimates, year,  "Solar")
    wind_estimated_data = estimate_wacc(CoCModel, wind_estimates, year, "Onshore_Wind")
    
    # Aggregated estimated data
    extracted_data = pd.merge(solar_estimated_data, wind_estimated_data, how="left", on="Country code", suffixes=("", "_y"))
    extracted_data.drop([col for col in extracted_data.columns if col.endswith('_y')],axis=1,inplace=True)
    
    # Extract waccs for onshore, offshore and solar PV
    estimated_country_waccs = pd.merge(CoCModel.country_mapping, extracted_data[['Country code', 'Onshore_Wind_WACC_' + year, "Solar_WACC_" + year]], how="left", on="Country code")
    estimated_country_waccs = estimated_country_waccs.rename(columns={"Onshore_Wind_WACC_" + year: "onshore_wacc", "Solar_WACC_" + year: "solar_pv_wacc"})
    
    
    return solar_estimated_data, estimated_country_waccs


    
CoCModel = wacc_model.CoCModel_class
irena_premiums = CoCModel.estimate_irena_premiums()
solar_estimates, estimated_waccs = estimate_future_waccs_v4(CoCModel,  "Onshore_Wind_Common_Risk_2021", "Solar_Common_Risk_2021", ['Wind_Penetration_2021', "CRP_2021", "RISE_OVERALL_2021"], ['Solar_Penetration_2021', "CRP_2021","RISE_OVERALL_2021"], "2023", 3.96)




solar = solar_results 
wind = wind_results
data = wacc_model.calculated_data

land_cover = xr.open_dataset("./DATA/GlobalLandCover.nc")
land_mapping = pd.read_csv("./DATA/LandUseCSV.csv")

def get_utilisations(land_cover, land_mapping, annual_production, technology):
        
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values
    global_cover = land_cover.reindex_like(annual_production, method="nearest")
    mapping = land_mapping

    utilisation = xr.zeros_like(global_cover['cover'])
    for i in np.arange(0, 21, 1):
        # Use xarray's where and isin functions to map land use categories to values
        if technology == "Solar":
            utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['PV LU'].iloc[i], utilisation)
        elif technology =="Wind":
            utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['Wind LU'].iloc[i], utilisation)
            utilisation = xr.where(np.isnan(self.PEM_100['levelised_cost'].sel(latitude=slice(-65, 90))) & ~np.isnan(self.PEM_0['levelised_cost'].sel(latitude=slice(-65, 90))), 1, utilisation)      

    return utilisation    


def get_areas(annual_production):
        
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values

    # Add an extra value to latitude and longitude coordinates
    latitudes_extended = np.append(latitudes, latitudes[-1] + np.diff(latitudes)[-1])
    longitudes_extended = np.append(longitudes, longitudes[-1] + np.diff(longitudes)[-1])

    # Calculate the differences between consecutive latitude and longitude points
    dlat_extended = np.diff(latitudes_extended)
    dlon_extended = np.diff(longitudes_extended)

    # Calculate the Earth's radius in kms
    radius = 6371

    # Compute the mean latitude value for each grid cell
    mean_latitudes_extended = (latitudes_extended[:-1] + latitudes_extended[1:]) / 2
    mean_latitudes_2d = mean_latitudes_extended[:, np.newaxis]

    # Convert the latitude differences and longitude differences from degrees to radians
    dlat_rad_extended = np.radians(dlat_extended)
    dlon_rad_extended = np.radians(dlon_extended)

    # Compute the area of each grid cell using the Haversine formula
    areas_extended = np.outer(dlat_rad_extended, dlon_rad_extended) * (radius ** 2) * np.cos(np.radians(mean_latitudes_2d))

    # Create a dataset with the three possible capital expenditures 
    area_dataset = xr.Dataset()
    area_dataset['latitude'] = latitudes
    area_dataset['longitude'] = longitudes
    area_dataset['area'] = (['latitude', 'longitude'], areas_extended, {'latitude': latitudes, 'longitude': longitudes})

    return area_dataset


def get_supply_curves(land_cover, land_mapping, levelised_costs, annual_production, waccs, technology, plot_global=None):


    # Calculate area of each grid point in kms 
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values
    grid_areas = get_areas(annual_production)
    utilisation_factors = get_utilisations(land_cover, land_mapping, annual_production, technology)

    # Set out constants
    if technology == "Wind":
        power_density = 6520 # kW/km2
    elif technology == "Solar PV":
        power_density = 32950  # kW/km2
    elif technology == "Hybrid":
        power_density = 6520 + solar_fractions * (32950 - 6520)
    installed_capacity = 1000

    # Scale annual hydrogen production by turbine density
    max_installed_capacity = power_density * grid_areas['area'] * utilisation_factors
    ratios = max_installed_capacity / installed_capacity
    technical_potential = annual_production * ratios

    # Create new dataset with cost and production volume
    data_vars = {'WACC': waccs, 'technical_potential': technical_potential,
                 'levelised_cost': levelised_costs}
    coords = {'latitude': latitudes,
              'longitude': longitudes}
    supply_curve_ds = xr.Dataset(data_vars=data_vars, coords=coords)

    return supply_curve_ds



solar_supply_curve = get_supply_curves(land_cover, land_mapping, solar_results['Calculated_LCOE'], solar_results['electricity_production'], solar_results['Estimated_WACC'], "Solar PV")












import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

solar = solar_results 
wind = wind_results
data = wacc_model.calculated_data

land_cover = xr.open_dataset("./DATA/GlobalLandCover.nc")
land_mapping = pd.read_csv("./DATA/LandUseCSV.csv")
country_grids = xr.open_dataset("./DATA/country_grids.nc")
country_grids['country'] = xr.where(np.isnan(country_grids['land']), country_grids['sea'], country_grids['land'])
country_mapping = pd.read_csv("./DATA/country_mapping.csv")
GDP = pd.read_csv("./DATA/GDPPerCapita.csv")[['Country code', '2022']]
GDP_country_mapping = pd.merge(country_mapping, GDP, on="Country code", how="left")

def get_utilisations(land_cover, land_mapping, annual_production, technology):
        
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values
    global_cover = land_cover.reindex_like(annual_production, method="nearest")
    mapping = land_mapping

    utilisation = xr.zeros_like(global_cover['cover'])
    for i in np.arange(0, 21, 1):
        # Use xarray's where and isin functions to map land use categories to values
        if technology == "Solar":
            utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['PV LU'].iloc[i], utilisation)
        elif technology =="Wind":
            utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['Wind LU'].iloc[i], utilisation)   

    return utilisation    


def get_areas(annual_production):
        
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values

    # Add an extra value to latitude and longitude coordinates
    latitudes_extended = np.append(latitudes, latitudes[-1] + np.diff(latitudes)[-1])
    longitudes_extended = np.append(longitudes, longitudes[-1] + np.diff(longitudes)[-1])

    # Calculate the differences between consecutive latitude and longitude points
    dlat_extended = np.diff(latitudes_extended)
    dlon_extended = np.diff(longitudes_extended)

    # Calculate the Earth's radius in kms
    radius = 6371

    # Compute the mean latitude value for each grid cell
    mean_latitudes_extended = (latitudes_extended[:-1] + latitudes_extended[1:]) / 2
    mean_latitudes_2d = mean_latitudes_extended[:, np.newaxis]

    # Convert the latitude differences and longitude differences from degrees to radians
    dlat_rad_extended = np.radians(dlat_extended)
    dlon_rad_extended = np.radians(dlon_extended)

    # Compute the area of each grid cell using the Haversine formula
    areas_extended = np.outer(dlat_rad_extended, dlon_rad_extended) * (radius ** 2) * np.cos(np.radians(mean_latitudes_2d))

    # Create a dataset with the three possible capital expenditures 
    area_dataset = xr.Dataset()
    area_dataset['latitude'] = latitudes
    area_dataset['longitude'] = longitudes
    area_dataset['area'] = (['latitude', 'longitude'], areas_extended, {'latitude': latitudes, 'longitude': longitudes})

    return area_dataset


def get_supply_curves(land_cover, land_mapping, levelised_costs, uniform_costs, annual_production, waccs, technology, country_grids, plot_global=None):


    # Calculate area of each grid point in kms 
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values
    grid_areas = get_areas(annual_production)
    utilisation_factors = get_utilisations(land_cover, land_mapping, annual_production, technology)

    # Set out constants
    if technology == "Wind":
        power_density = 6520 # kW/km2
    elif technology == "Solar":
        power_density = 32950  # kW/km2
    elif technology == "Hybrid":
        power_density = 6520 + solar_fractions * (32950 - 6520)
    installed_capacity = 1000

    # Scale annual hydrogen production by turbine density
    max_installed_capacity = power_density * grid_areas['area'] * utilisation_factors
    ratios = max_installed_capacity / installed_capacity
    technical_potential = annual_production * ratios

    # Create new dataset with cost and production volume
    # Add in country level data
    data_vars = {'WACC': waccs, 'technical_potential': technical_potential,
                 'Calculated_LCOE': levelised_costs, 'Uniform_LCOE': uniform_costs, 'country': country_grids['country']}
    coords = {'latitude': latitudes,
              'longitude': longitudes}
    supply_curve_ds = xr.Dataset(data_vars=data_vars, coords=coords)

    return supply_curve_ds





def produce_wacc_potential_curve(supply_ds, GDP_country_mapping):
    
    def clean_results(dataframe):
        # Sort by levelised cost
        sorted_supply = dataframe.sort_values(by=['WACC'])

        # Remove rows that are empty
        cleaned_df = sorted_supply.dropna(axis='index')
        final_df = cleaned_df.copy()

        # Apply a threshold for locations with zero utilisation (if applicable)
        util_index_names = final_df[final_df['technical_potential'] == 0 ].index
        final_df.drop(util_index_names, inplace = True)

        return final_df
    
    # Convert the dataset to a dataframe
    supply_df = supply_ds.sel(year=2022).to_dataframe()

    # Extract the WACC and technical potential
    cleaned_results_df = clean_results(supply_df)

    # Plot the results
    sorted_lc = cleaned_results_df.sort_values(by=['WACC'], ascending=True)
    sorted_lc.loc[:, 'cumulative_potential'] = sorted_lc['technical_potential'].cumsum()
    rounded_df = sorted_lc.round({'WACC': 2})
    rounded_df = rounded_df.sort_values(by=['WACC'], ascending=True)
    fig, ax = plt.subplots(figsize=(20, 8))
    color_labels = {}
    cmap = mpl.colormaps['gnuplot_r']
    norm = mpl.colors.Normalize(vmin=0, vmax=50000)  # Normalize to the range of solar fractions
    

    

    # Iterate through each data point and create a bar with the specified width
    for index, row in rounded_df.iterrows():
        width = row['technical_potential'] / 1e+09  # Bar width
        height = row['WACC']  # Bar height
        country = row['country']
        cumulative_production = row['cumulative_potential'] / 1e+09  # Cumulative production
        
        # Get GDP per capita
        gdp_per_capita = GDP_country_mapping.loc[GDP_country_mapping['index'] == country, '2022'].values[0]
        color = cmap(norm(gdp_per_capita))

        # Plot a bar with the specified width, height, x-position, and color
        ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=color)
        
    
    def thousands_format(x, pos):
        return f'{int(x):,}'
    
    # Set labels
    ax.set_xlim(0, cumulative_production)
    ax.set_ylim(0, 25)
    ax.set_ylabel('WACC (%)', fontsize=20)
    ax.set_xlabel('Annual Electricity Potential (TWh/year)', fontsize=25)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_format))

    # Set the size of x and y-axis tick labels
    ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
    ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed
    
    # Add color bar
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=[0, 5000, 10000, 20000, 25000, 50000], format=',', extend="max", anchor=(0.25, 0.5))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(thousands_format))
    cbar.set_label('GDP per capita (USD, 2022)', fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    
    # Get the LCOE plot
    lcoe_data = cleaned_results_df[['Calculated_LCOE', 'Uniform_LCOE', 'technical_potential']]
    sorted_lcoe = cleaned_results_df.sort_values(by=['Calculated_LCOE'], ascending=True).copy()
    sorted_lcoe.loc[:, 'cumulative_potential'] = sorted_lcoe['technical_potential'].cumsum()
    sorted_uniform_lcoe = cleaned_results_df.sort_values(by=['Uniform_LCOE'], ascending=True).copy()
    sorted_uniform_lcoe.loc[:, 'cumulative_potential'] = sorted_uniform_lcoe['technical_potential'].cumsum()
    
    
    # Plot second axis

    # Create a twin Y-axis on the same figure
    ax2 = ax.twinx()

    # Plot lines on the twin Y-axis (this axis is independent of the main y-axis)
    ax2.plot(sorted_lcoe['cumulative_potential']/1e+9, sorted_lcoe['Calculated_LCOE']*1000, color='blue', lw=2.5, label='LCOE under Country-level WACC', linestyle="--")
    ax2.plot(sorted_uniform_lcoe['cumulative_potential']/1e+9, sorted_uniform_lcoe['Uniform_LCOE']*1000, color='red', lw=2.5, label='LCOE under Uniform 7% WACC', linestyle="--")

    # Customize the second y-axis
    ax2.set_ylabel('Levelised Cost (USD/MWh)', fontsize=20)
    ax2.legend(loc='upper left', fontsize=20)
    ax2.tick_params(axis="y", labelsize=20)
    ax2.set_ylim(0, 250)

    plt.show()
    
    return rounded_df
    
solar_supply_curve = get_supply_curves(land_cover, land_mapping, solar_results['Calculated_LCOE'], solar_results['Uniform_LCOE'], solar_results['electricity_production'], solar_results['Estimated_WACC'], "Solar", country_grids)
solar_df = produce_wacc_potential_curve(solar_supply_curve, GDP_country_mapping)

wind_supply_curve = get_supply_curves(land_cover, land_mapping, wind_results['Calculated_LCOE'], wind_results['Uniform_LCOE'], wind_results['electricity_production'], wind_results['Estimated_WACC'], "Wind", country_grids)
wind_df = produce_wacc_potential_curve(wind_supply_curve, GDP_country_mapping)





import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

solar = solar_results 
wind = wind_results
data = wacc_model.calculated_data

land_cover = xr.open_dataset("./DATA/GlobalLandCover.nc")
land_mapping = pd.read_csv("./DATA/LandUseCSV.csv")
country_grids = xr.open_dataset("./DATA/country_grids.nc")
country_grids['country'] = xr.where(np.isnan(country_grids['land']), country_grids['sea'], country_grids['land'])
country_mapping = pd.read_csv("./DATA/country_mapping.csv")
GDP = pd.read_csv("./DATA/GDPPerCapita.csv")[['Country code', '2022']]
GDP_country_mapping = pd.merge(country_mapping, GDP, on="Country code", how="left")

def get_utilisations(land_cover, land_mapping, annual_production, technology):
        
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values
    global_cover = land_cover.reindex_like(annual_production, method="nearest")
    mapping = land_mapping

    utilisation = xr.zeros_like(global_cover['cover'])
    for i in np.arange(0, 21, 1):
        # Use xarray's where and isin functions to map land use categories to values
        if technology == "Solar":
            utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['PV LU'].iloc[i], utilisation)
        elif technology =="Wind":
            utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['Wind LU'].iloc[i], utilisation)   

    return utilisation    


def get_areas(annual_production):
        
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values

    # Add an extra value to latitude and longitude coordinates
    latitudes_extended = np.append(latitudes, latitudes[-1] + np.diff(latitudes)[-1])
    longitudes_extended = np.append(longitudes, longitudes[-1] + np.diff(longitudes)[-1])

    # Calculate the differences between consecutive latitude and longitude points
    dlat_extended = np.diff(latitudes_extended)
    dlon_extended = np.diff(longitudes_extended)

    # Calculate the Earth's radius in kms
    radius = 6371

    # Compute the mean latitude value for each grid cell
    mean_latitudes_extended = (latitudes_extended[:-1] + latitudes_extended[1:]) / 2
    mean_latitudes_2d = mean_latitudes_extended[:, np.newaxis]

    # Convert the latitude differences and longitude differences from degrees to radians
    dlat_rad_extended = np.radians(dlat_extended)
    dlon_rad_extended = np.radians(dlon_extended)

    # Compute the area of each grid cell using the Haversine formula
    areas_extended = np.outer(dlat_rad_extended, dlon_rad_extended) * (radius ** 2) * np.cos(np.radians(mean_latitudes_2d))

    # Create a dataset with the three possible capital expenditures 
    area_dataset = xr.Dataset()
    area_dataset['latitude'] = latitudes
    area_dataset['longitude'] = longitudes
    area_dataset['area'] = (['latitude', 'longitude'], areas_extended, {'latitude': latitudes, 'longitude': longitudes})

    return area_dataset


def get_supply_curves(land_cover, land_mapping, levelised_costs, uniform_costs, annual_production, waccs, technology, country_grids, plot_global=None):


    # Calculate area of each grid point in kms 
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values
    grid_areas = get_areas(annual_production)
    utilisation_factors = get_utilisations(land_cover, land_mapping, annual_production, technology)

    # Set out constants
    if technology == "Wind":
        power_density = 6520 # kW/km2
    elif technology == "Solar":
        power_density = 32950  # kW/km2
    elif technology == "Hybrid":
        power_density = 6520 + solar_fractions * (32950 - 6520)
    installed_capacity = 1000

    # Scale annual hydrogen production by turbine density
    max_installed_capacity = power_density * grid_areas['area'] * utilisation_factors
    ratios = max_installed_capacity / installed_capacity
    technical_potential = annual_production * ratios

    # Create new dataset with cost and production volume
    # Add in country level data
    data_vars = {'WACC': waccs, 'technical_potential': technical_potential,
                 'Calculated_LCOE': levelised_costs, 'Uniform_LCOE': uniform_costs, 'country': country_grids['country']}
    coords = {'latitude': latitudes,
              'longitude': longitudes}
    supply_curve_ds = xr.Dataset(data_vars=data_vars, coords=coords)

    return supply_curve_ds





def produce_wacc_potential_curve(supply_ds, GDP_country_mapping, filename=None, graphmarking=None, title=None):
    
    def clean_results(dataframe):
        # Sort by levelised cost
        sorted_supply = dataframe.sort_values(by=['WACC'])

        # Remove rows that are empty
        cleaned_df = sorted_supply.dropna(axis='index')
        final_df = cleaned_df.copy()

        # Apply a threshold for locations with zero utilisation (if applicable)
        util_index_names = final_df[final_df['technical_potential'] == 0 ].index
        final_df.drop(util_index_names, inplace = True)

        return final_df
    
    # Convert the dataset to a dataframe
    supply_df = supply_ds.sel(year=2022).to_dataframe()

    # Extract the WACC and technical potential
    cleaned_results_df = clean_results(supply_df)

    # Plot the results
    sorted_lc = cleaned_results_df.sort_values(by=['WACC'], ascending=True)
    sorted_lc.loc[:, 'cumulative_potential'] = sorted_lc['technical_potential'].cumsum()
    rounded_df = sorted_lc.round({'WACC': 2})
    rounded_df = rounded_df.sort_values(by=['WACC'], ascending=True)
    fig, ax = plt.subplots(figsize=(20, 8))
    color_labels = {}
    cmap = mpl.colormaps['gnuplot_r']
    norm = mpl.colors.Normalize(vmin=0, vmax=50000)  # Normalize to the range of solar fractions
    

    

    # Iterate through each data point and create a bar with the specified width
    for index, row in rounded_df.iterrows():
        width = row['technical_potential'] / 1e+09  # Bar width
        height = row['WACC']  # Bar height
        country = row['country']
        cumulative_production = row['cumulative_potential'] / 1e+09  # Cumulative production
        
        # Get GDP per capita
        gdp_per_capita = GDP_country_mapping.loc[GDP_country_mapping['index'] == country, '2022'].values[0]
        color = cmap(norm(gdp_per_capita))

        # Plot a bar with the specified width, height, x-position, and color
        ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=color)
        
    
    def thousands_format(x, pos):
        return f'{int(x):,}'
    
    # Set labels
    ax.set_xlim(0, 150000)
    ax.set_ylim(0, 25)
    ax.set_ylabel('WACC (%)', fontsize=20)
    ax.set_xlabel('Annual Electricity Potential (TWh/year)', fontsize=25)
    ax.set_title(title, fontsize=30)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_format))

    # Set the size of x and y-axis tick labels
    ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
    ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed
    
    # Add color bar
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 450000,50000], format=',', extend="max", anchor=(0.25, 0.5))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(thousands_format))
    cbar.set_label('GDP per capita (USD, 2022)', fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    
    # Get the LCOE plot
    lcoe_data = cleaned_results_df[['Calculated_LCOE', 'Uniform_LCOE', 'technical_potential']]
    sorted_lcoe = cleaned_results_df.sort_values(by=['Calculated_LCOE'], ascending=True).copy()
    sorted_lcoe.loc[:, 'cumulative_potential'] = sorted_lcoe['technical_potential'].cumsum()
    sorted_uniform_lcoe = cleaned_results_df.sort_values(by=['Uniform_LCOE'], ascending=True).copy()
    sorted_uniform_lcoe.loc[:, 'cumulative_potential'] = sorted_uniform_lcoe['technical_potential'].cumsum()
    
    
    # Plot second axis
    # Create a twin Y-axis on the same figure
    ax2 = ax.twinx()

    # Plot lines on the twin Y-axis (this axis is independent of the main y-axis)
    ax2.plot(sorted_lcoe['cumulative_potential']/1e+9, sorted_lcoe['Calculated_LCOE']*1000, color='blue', lw=2.5, label='LCOE under Country-level WACC', linestyle="--")
    ax2.plot(sorted_uniform_lcoe['cumulative_potential']/1e+9, sorted_uniform_lcoe['Uniform_LCOE']*1000, color='red', lw=2.5, label='LCOE under Uniform 7% WACC', linestyle="--")

    # Customize the second y-axis
    ax2.set_ylabel('Levelised Cost (USD/MWh)', fontsize=20)
    ax2.legend(loc='upper center', fontsize=20)
    ax2.tick_params(axis="y", labelsize=20)
    ax2.set_ylim(0, 250)
    
    if graphmarking is not None:
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
    
    if filename is not None:
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()
    
    return rounded_df


def produce_lcoe_potential_curve(supply_ds, region_mapping, filename=None, graphmarking=None, title=None):
    

        # Iterate through each data point and create a bar with the specified width
        for index, row in rounded_df.iterrows():
            width = row['hydrogen_technical_potential'] / 1e+06  # Bar width
            height = row['levelised_cost']  # Bar height
            solar_fraction = row['Optimal_SF']
            solar_frac_label = str(int(solar_fraction))
            cumulative_production = row['total_cumulative_potential'] / 1e+06  # Cumulative production
            color = cmap(solar_fraction)

            # Create a dummy bar element with the color and label
            dummy_bar = plt.bar([], [], color=color, label=solar_frac_label)

            # Add the color and label to the dictionary
            color_labels[color] = solar_frac_label

            # Plot a bar with the specified width, height, x-position, and color
            ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=color)

        # Set labels
        ax.set_xlim(0, cumulative_production)
        ax.set_ylabel('LCOH(US$/kg)', fontsize=25)
        ax.set_xlabel('Annual Hydrogen Production (Mt/a)', fontsize=25)
        ax.set_ylim([0, 10])
        ax.plot(rounded_df['total_cumulative_potential'], rounded_df['levelised_cost'], linewidth=10, color='black')

        # Set the size of x and y-axis tick labels
        ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
        ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed

        # Create a legend based on the color_labels dictionary
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{color_labels[color]}',
                                  markerfacecolor=color, markersize=20) for color in color_labels]

        #plt.legend(handles=legend_elements, title='Solar Fraction', loc='upper left', fontsize=15, title_fontsize=20, ncol=4)

        # Add color bar
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Solar Fraction (%)', fontsize=20)
        cbar.ax.tick_params(labelsize=15)

        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
        plt.savefig(filename + ".png")
        plt.show()
    
    
    
    
    def clean_results(dataframe):
        # Sort by levelised cost
        sorted_supply = dataframe.sort_values(by=['WACC'])

        # Remove rows that are empty
        cleaned_df = sorted_supply.dropna(axis='index')
        final_df = cleaned_df.copy()

        # Apply a threshold for locations with zero utilisation (if applicable)
        util_index_names = final_df[final_df['technical_potential'] == 0 ].index
        final_df.drop(util_index_names, inplace = True)

        return final_df
    
    # Convert the dataset to a dataframe
    supply_df = supply_ds.sel(year=2022).to_dataframe()

    # Extract the WACC and technical potential
    cleaned_results_df = clean_results(supply_df)

    # Plot the results
    sorted_lc = cleaned_results_df.sort_values(by=['Calculated_LCOE'], ascending=True)
    sorted_lc.loc[:, 'cumulative_potential'] = sorted_lc['technical_potential'].cumsum()
    rounded_df = sorted_lc.round({'Calculated_LCOE': 2})
    rounded_df = rounded_df.sort_values(by=['Calculated_LCOE'], ascending=True)
    fig, ax = plt.subplots(figsize=(20, 8))
    region_hatching = dict(('/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*')
    cmap = mpl.colormaps['gnuplot_r']
    norm = mpl.colors.Normalize(vmin=0, vmax=50000)  # Normalize to the range of solar fractions    

    # Iterate through each data point and create a bar with the specified width
    for index, row in rounded_df.iterrows():
        width = row['technical_potential'] / 1e+09  # Bar width
        height = row['WACC']  # Bar height
        country = row['country']
        cumulative_production = row['cumulative_potential'] / 1e+09  # Cumulative production
        
        # Get GDP per capita
        gdp_per_capita = GDP_country_mapping.loc[GDP_country_mapping['index'] == country, '2022'].values[0]
        color = cmap(norm(gdp_per_capita))

        # Plot a bar with the specified width, height, x-position, and color
        ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=color)
        
    
    def thousands_format(x, pos):
        return f'{int(x):,}'
    
    

    # Iterate through each data point and create a bar with the specified width
    for index, row in rounded_df.iterrows():
        width = row['technical_potential'] / 1e+09  # Bar width
        height = row['Calculated_LCOE']  # Bar height
        country = row['country']
        cumulative_production = row['cumulative_potential'] / 1e+09  # Cumulative production
        
        # Get GDP per capita
        region_color = region_mapping.loc[region_mapping['index'] == country, 'Region Color'].values[0]
        region_label = region_mapping.loc[region_mapping['index'] == country, 'Region'].values[0]
        
        # Create a dummy bar element with the color and label
        dummy_bar = plt.bar([], [], color=color, label=region_label)

        # Plot a bar with the specified width, height, x-position, and color
        ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=region_color)
        
    
    def thousands_format(x, pos):
        return f'{int(x):,}'
    
    # Set labels
    ax.set_xlim(0, 150000)
    ax.set_ylim(0, 25)
    ax.set_ylabel('Levelised Cost of Electricity (LCOE)', fontsize=20)
    ax.set_xlabel('Annual Electricity Potential (TWh/year)', fontsize=25)
    ax.set_title(title, fontsize=30)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_format))

    # Set the size of x and y-axis tick labels
    ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
    ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed
    
    # Add color bar
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 450000,50000], format=',', extend="max", anchor=(0.25, 0.5))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(thousands_format))
    cbar.set_label('GDP per capita (USD, 2022)', fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    
    # Get the LCOE plot
    lcoe_data = cleaned_results_df[['Calculated_LCOE', 'Uniform_LCOE', 'technical_potential']]
    sorted_lcoe = cleaned_results_df.sort_values(by=['Calculated_LCOE'], ascending=True).copy()
    sorted_lcoe.loc[:, 'cumulative_potential'] = sorted_lcoe['technical_potential'].cumsum()
    sorted_uniform_lcoe = cleaned_results_df.sort_values(by=['Uniform_LCOE'], ascending=True).copy()
    sorted_uniform_lcoe.loc[:, 'cumulative_potential'] = sorted_uniform_lcoe['technical_potential'].cumsum()
    
    
    # Plot second axis
    # Create a twin Y-axis on the same figure
    ax2 = ax.twinx()

    # Plot lines on the twin Y-axis (this axis is independent of the main y-axis)
    ax2.plot(sorted_lcoe['cumulative_potential']/1e+9, sorted_lcoe['Calculated_LCOE']*1000, color='blue', lw=2.5, label='LCOE under Country-level WACC', linestyle="--")
    ax2.plot(sorted_uniform_lcoe['cumulative_potential']/1e+9, sorted_uniform_lcoe['Uniform_LCOE']*1000, color='red', lw=2.5, label='LCOE under Uniform 7% WACC', linestyle="--")

    # Customize the second y-axis
    ax2.set_ylabel('Levelised Cost (USD/MWh)', fontsize=20)
    ax2.legend(loc='upper center', fontsize=20)
    ax2.tick_params(axis="y", labelsize=20)
    ax2.set_ylim(0, 250)
    
    if graphmarking is not None:
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
    
    if filename is not None:
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()
    
    return rounded_df

    
solar_supply_curve = get_supply_curves(land_cover, land_mapping, solar_results['Calculated_LCOE'], solar_results['Uniform_LCOE'], solar_results['electricity_production'], solar_results['Estimated_WACC'], "Solar", country_grids)
solar_df = produce_wacc_potential_curve(solar_supply_curve, GDP_country_mapping, filename="SolarGlobalSupplyCurve", graphmarking="a", title="Solar")

wind_supply_curve = get_supply_curves(land_cover, land_mapping, wind_results['Calculated_LCOE'], wind_results['Uniform_LCOE'], wind_results['electricity_production'], wind_results['Estimated_WACC'], "Wind", country_grids)
wind_df = produce_wacc_potential_curve(wind_supply_curve, GDP_country_mapping, filename="WindGlobalSupplyCurve", graphmarking="b", title="Wind")
                            
                            

def plot_wacc_values(estimated_waccs, country_mapping, technology): 
    
    data = country_mapping
    storage_df = xr.zeros_like(data['land'])
    if technology == "Offshore Wind":
        storage_df = xr.zeros_like(data['sea'])
    storage_df = xr.where(storage_df == 0, np.nan, np.nan)
    for i in np.arange(1, 251, 1):
        # Extract WACC
        if technology == "Solar":
            wacc = estimated_waccs[estimated_waccs['index'] == i]['solar_pv_wacc'].values[0] 
        elif technology == "Onshore Wind":
            wacc = estimated_waccs[estimated_waccs['index'] == i]['onshore_wacc'].values[0] 
        elif technology == "Offshore Wind":
            wacc = estimated_waccs[estimated_waccs['index'] == i]['offshore_wacc'].values[0] 
            
        # Apply mapping
        storage_df = xr.where(data['land'] == i, wacc, storage_df)
        
    # Extracted data
    extracted_data = storage_df
    
    return extracted_data

country_geodata = wacc_model.geodata.reindex({"latitude":solar_results.latitude, "longitude":solar_results.longitude}, method="nearest")


solar_plot_waccs = plot_wacc_values(wacc_model.country_wacc_mapping, country_geodata, "Solar")
solar_plot_waccs = xr.where(np.isnan(solar_results['Calculated_LCOE']), np.nan, solar_plot_waccs)
onshore_plot_waccs = plot_wacc_values(wacc_model.country_wacc_mapping, country_geodata, "Onshore Wind")
onshore_plot_waccs = xr.where(np.isnan(solar_results['Calculated_LCOE']), np.nan, onshore_plot_waccs)
offshore_plot_waccs = plot_wacc_values(wacc_model.country_wacc_mapping, country_geodata, "Offshore Wind")
offshore_plot_waccs = xr.where(np.isnan(onshore_plot_waccs), np.nan, offshore_plot_waccs)

wacc_model.plot_data_shading(solar_plot_waccs,solar_plot_waccs.latitude, solar_plot_waccs.longitude, tick_values = [0, 5, 10, 15, 20], title="Estimated\nWACC \n (%, real,\nafter tax)\n", cmap="YlOrRd", extend_set="neither", filename = output_folder + "Solar_WACC_2023", graphmarking="a")

wacc_model.plot_data_shading(onshore_plot_waccs, onshore_plot_waccs.latitude, onshore_plot_waccs.longitude, tick_values = [0, 5, 10, 15, 20], title="Estimated\nWACC\n (%, real,\nafter tax)\n", cmap="YlGnBu", extend_set="neither", filename = output_folder + "Wind_WACC_2023", graphmarking="b")

wacc_model.plot_data_shading(offshore_plot_waccs, offshore_plot_waccs.latitude, offshore_plot_waccs.longitude, tick_values = [0, 5, 10, 15, 20], title="Estimated\nWACC\n (%, real,\nafter tax)\n", cmap="YlGnBu", extend_set="neither", filename = output_folder + "Offshore_Wind_WACC_2023", graphmarking="c")
                            
                            
                            
                            
                            
                            
                            
                            
def get_utilisations(land_cover, land_mapping, annual_production, technology):

    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values
    global_cover = land_cover.reindex_like(annual_production, method="nearest")
    mapping = land_mapping

    utilisation = xr.zeros_like(global_cover['cover'])
    for i in np.arange(0, 21, 1):
        # Use xarray's where and isin functions to map land use categories to values
        if technology == "Solar":
            utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['PV LU'].iloc[i], utilisation)
        elif technology =="Wind":
            utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['Wind LU'].iloc[i], utilisation)   

    return utilisation    


def get_areas(annual_production):

    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values

    # Add an extra value to latitude and longitude coordinates
    latitudes_extended = np.append(latitudes, latitudes[-1] + np.diff(latitudes)[-1])
    longitudes_extended = np.append(longitudes, longitudes[-1] + np.diff(longitudes)[-1])

    # Calculate the differences between consecutive latitude and longitude points
    dlat_extended = np.diff(latitudes_extended)
    dlon_extended = np.diff(longitudes_extended)

    # Calculate the Earth's radius in kms
    radius = 6371

    # Compute the mean latitude value for each grid cell
    mean_latitudes_extended = (latitudes_extended[:-1] + latitudes_extended[1:]) / 2
    mean_latitudes_2d = mean_latitudes_extended[:, np.newaxis]

    # Convert the latitude differences and longitude differences from degrees to radians
    dlat_rad_extended = np.radians(dlat_extended)
    dlon_rad_extended = np.radians(dlon_extended)

    # Compute the area of each grid cell using the Haversine formula
    areas_extended = np.outer(dlat_rad_extended, dlon_rad_extended) * (radius ** 2) * np.cos(np.radians(mean_latitudes_2d))

    # Create a dataset with the three possible capital expenditures 
    area_dataset = xr.Dataset()
    area_dataset['latitude'] = latitudes
    area_dataset['longitude'] = longitudes
    area_dataset['area'] = (['latitude', 'longitude'], areas_extended, {'latitude': latitudes, 'longitude': longitudes})

    return area_dataset



def get_supply_curves_v2(data, land_cover, land_mapping, technology, country_grids):
    
    # Extract required parameters
    annual_production = data['electricity_production']
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values
    
    # Get area and utilisations
    grid_areas = get_areas(annual_production)
    utilisation_factors = get_utilisations(land_cover, land_mapping, annual_production, technology)
    
    # Set out constants
    if technology == "Wind":
        power_density = 6520 # kW/km2
    elif technology == "Solar":
        power_density = 32950  # kW/km2
    installed_capacity = 1000
    
    # Scale annual electricity production by power density
    max_installed_capacity = power_density * grid_areas['area'] * utilisation_factors
    ratios = max_installed_capacity / installed_capacity
    technical_potential = annual_production * ratios
    
    # Include additional data into the dataset
    data['technical_potential'] = technical_potential
    data['country'] = country_grids['country']
    
    return data

def produce_wacc_potential_curve_v2(supply_ds, GDP_country_mapping, filename=None, graphmarking=None, title=None, xlim=None):
    
    # Convert the dataset into a dataframe
    supply_df = supply_ds.to_dataframe()
    
    # Remove locations not evaluated
    supply_df = supply_df.dropna(axis="index")
    
    # Create two copies
    uniform_df = supply_ds.to_dataframe().dropna(subset="Uniform_LCOE")
    wacc_df = supply_ds.to_dataframe().dropna(subset="Calculated_LCOE")
    
    # For the Country WACC case, sort values and calculate cumulative sum
    supply_df = supply_df.round({'Estimated_WACC': 3})
    supply_df = supply_df.sort_values(by=['Estimated_WACC'], ascending=True)
    supply_df['cumulative_potential'] = supply_df['technical_potential'].cumsum()
    
    # For the WACC case, sort values and calculate cumulative sum
    wacc_df = wacc_df.sort_values(by=['Calculated_LCOE'], ascending=True)
    wacc_df['cumulative_wacc'] = wacc_df['technical_potential'].cumsum()
    
    # For the Uniform WACC case, sort values and calculate cumulative sum
    uniform_df = uniform_df.sort_values(by=['Uniform_LCOE'], ascending=True)
    uniform_df['cumulative_uniform'] = uniform_df['technical_potential'].cumsum()
    
    # Plot test checker
    plt.plot(wacc_df['cumulative_wacc'], wacc_df['Calculated_LCOE'])
    plt.plot(uniform_df['cumulative_uniform'], uniform_df['Uniform_LCOE'])
    plt.ylim(0, 0.25)


    # Plot the results
    fig, ax = plt.subplots(figsize=(20, 8))
    color_labels = {}
    cmap = mpl.colormaps['gnuplot_r']
    norm = mpl.colors.Normalize(vmin=0, vmax=50000)  # Normalize to the range of GDP


    # Iterate through each data point and create a bar with the specified width
    for index, row in supply_df.iterrows():
        width = row['technical_potential'] / 1e+09  # Bar width, in TWh
        height = row['Estimated_WACC']  # Bar height
        country = row['country']
        cumulative_production = row['cumulative_potential'] / 1e+09  # Cumulative production, in TWh

        # Get GDP per capita 
        if np.isnan(country):
            gdp_per_capita = np.nan
        else:
            gdp_per_capita = GDP_country_mapping.loc[GDP_country_mapping['index'] == country, '2022'].values[0]
        color = cmap(norm(gdp_per_capita))

        # Plot a bar with the specified width, height, x-position, and color
        ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=color)


    def thousands_format(x, pos):
        return f'{int(x):,}'

    # Set labels
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 25)
    ax.set_ylabel('WACC (%)', fontsize=20)
    ax.set_xlabel('Annual Electricity Potential (TWh/year)', fontsize=25)
    ax.set_title(title, fontsize=30)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_format))

    # Set the size of x and y-axis tick labels
    ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
    ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed

    # Add color bar
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 450000,50000], format=',', extend="max", anchor=(0.25, 0.5))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(thousands_format))
    cbar.set_label('GDP per capita (USD, 2022)', fontsize=20)
    cbar.ax.tick_params(labelsize=15)


    # Plot second axis
    # Create a twin Y-axis on the same figure
    ax2 = ax.twinx()

    # Plot lines on the twin Y-axis (this axis is independent of the main y-axis)
    ax2.plot(wacc_df['cumulative_wacc']/1e+9, wacc_df['Calculated_LCOE']*1000, color='blue', lw=2.5, label='LCOE under Country-level WACC', linestyle="--")
    ax2.plot(uniform_df['cumulative_uniform']/1e+9, uniform_df['Uniform_LCOE']*1000, color='red', lw=2.5, label='LCOE under Uniform 7% WACC', linestyle="--")

    # Customize the second y-axis
    ax2.set_ylabel('Levelised Cost (USD/MWh)', fontsize=20)
    ax2.legend(loc='upper center', fontsize=20)
    ax2.tick_params(axis="y", labelsize=20)
    ax2.set_ylim(0, 250)

    if graphmarking is not None:
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

    if filename is not None:
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()




GDP = pd.read_csv("./DATA/GDPPerCapita.csv")[['Country code', '2022']]
GDP_country_mapping = pd.merge(country_mapping, GDP, on="Country code", how="left")
solar_ds = get_supply_curves_v2(solar_results, land_cover, land_mapping, "Solar", country_grids)
produce_wacc_potential_curve_v2(solar_ds, GDP_country_mapping)


solar_df = solar_results.to_dataframe()
solar_df = solar_df.dropna(subset="Calculated_LCOE")
solar_df = solar_df.sort_values(by="Calculated_LCOE", ascending=True)
solar_df['cumulative_production'] = solar_df['technical_potential'].cumsum()
plt.plot(solar_df['cumulative_production'], solar_df['Calculated_LCOE'])

uniform_df = uniform_df.dropna(subset="Uniform_LCOE")
uniform_df = uniform_df.sort_values(by="Uniform_LCOE", ascending=True)
uniform_df['cumulative_uniform'] = uniform_df['technical_potential'].cumsum()
plt.plot(uniform_df['cumulative_uniform'], uniform_df['Uniform_LCOE'])
plt.ylim(0, 0.25)

wind_ds = get_supply_curves_v2(wind_results, land_cover, land_mapping, "Wind", country_grids)
produce_wacc_potential_curve_v2(wind_ds, GDP_country_mapping)
                           
                            
def get_concessionality_v3(wacc_model, data, concessional_rate, technology, risk_reductions):
    
    def reduce_30_percent(data, rf_rate, risk_reductions):
        
        # Further modify 'WACC_2023' only for the rows in the mask
        data.loc[data['Debt_Share_2023'] < 80, 'Debt_Cost_2023'] = (data.loc[data['Debt_Share_2023'] < 80, 'Debt_Cost_2023'] - rf_rate) * ((100 - risk_reductions) / 100) + rf_rate
        data.loc[data['Debt_Share_2023'] < 80, 'Equity_Cost_2023'] = (data.loc[data['Debt_Share_2023'] < 80, 'Equity_Cost_2023'] - rf_rate) * ((100 - risk_reductions) / 100) + rf_rate
        
        return data
    
    
    # Convert data in netcdf to pandas dataframe
    merged_dataset = xr.Dataset({
    "Required_WACC": data,
    "index": wacc_model.land_grids
})
    working_dataframe = merged_dataset.to_dataframe().reset_index()
    
    # Import the existing costs of commercial debt, equity and the modelled debt share
    if technology == "Wind":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Onshore_Wind_WACC_2023', 'Onshore_Wind_Cost_Debt_2023', 'Onshore_Wind_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Onshore_Wind_WACC_2023":"WACC_2023", "Onshore_Wind_Cost_Debt_2023": "Debt_Cost_2023", "Onshore_Wind_Cost_Equity_2023":"Equity_Cost_2023"})
    elif technology == "Solar":
        financing_values = wacc_model.calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Solar_WACC_2023', 'Solar_Cost_Debt_2023', 'Solar_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Solar_WACC_2023":"WACC_2023", "Solar_Cost_Debt_2023": "Debt_Cost_2023", "Solar_Cost_Equity_2023":"Equity_Cost_2023"})
    financing_terms = pd.merge(working_dataframe, financing_values, how="left", on="index") 
    
    # Reduce the overall cost of capital by 30% for developing countries
    financing_terms = reduce_30_percent(financing_terms, 3.96, risk_reductions)
    
    
    # Calculate debt-equity ratio
    financing_terms['Equity_Share_2023'] = (100 - financing_terms['Debt_Share_2023']) 
    financing_terms['Debt_Equity_Ratio'] =  financing_terms['Debt_Share_2023'] / financing_terms['Equity_Share_2023']
    
    # Set the cost of concessional financing 
    financing_terms['Concessional_Cost_2023'] = concessional_rate 
    
    
    # Calculate the share of concessional financing / commercial debt / commercial equity
    numerator =  (financing_terms['Debt_Equity_Ratio'] + 1) * financing_terms['Required_WACC'] - financing_terms['Debt_Cost_2023'] * financing_terms['Debt_Equity_Ratio'] * (1 - financing_terms['Tax_Rate']/100) - financing_terms['Equity_Cost_2023']
    denominator = ((financing_terms['Debt_Equity_Ratio'] + 1) * financing_terms['Concessional_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100) - financing_terms['Equity_Cost_2023'] - financing_terms['Debt_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100) * financing_terms['Debt_Equity_Ratio'])
    financing_terms['Concessional_Debt_Share'] = numerator / denominator
    financing_terms['Commercial_Equity_Share'] = (1 - financing_terms['Concessional_Debt_Share']) / (financing_terms['Debt_Equity_Ratio'] + 1) 
    financing_terms['Commercial_Debt_Share'] = 1 - financing_terms['Concessional_Debt_Share'] - financing_terms['Commercial_Equity_Share'] 
    
    # Calculate shares of final
    financing_terms['Equity_Contribution'] = financing_terms['Commercial_Equity_Share'] * (financing_terms['Equity_Cost_2023'])
    financing_terms['Debt_Contribution'] = financing_terms['Commercial_Debt_Share'] * (financing_terms['Debt_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100) ) 
    financing_terms['Concessional_Contribution'] = financing_terms['Concessional_Debt_Share'] * financing_terms['Concessional_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100)
    financing_terms['Concessionality'] = financing_terms['Debt_Cost_2023'] - financing_terms['Concessional_Cost_2023']
    
    # Create a check
    financing_terms['Total_Check'] =  financing_terms['Concessional_Debt_Share'] + financing_terms['Commercial_Debt_Share'] + financing_terms['Commercial_Equity_Share']
    financing_terms['WACC_Concessional'] =  financing_terms['Equity_Contribution'] + financing_terms['Debt_Contribution'] + financing_terms['Concessional_Contribution']
    
    # Address inequalities
    financing_terms['Concessional_Debt_Share'] = financing_terms['Concessional_Debt_Share'] * 100
    financing_terms.loc[financing_terms['Required_WACC'] == 999, "Concessional_Debt_Share"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] == 111, "Concessional_Debt_Share"] = 111
    financing_terms.loc[financing_terms['Required_WACC'] < concessional_rate, "Concessional_Debt_Share"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] < concessional_rate, "Commercial_Equity_Share"] = 999
    financing_terms.loc[financing_terms['Required_WACC'] < concessional_rate, "Commercial_Debt_Share"] = 999
    
    # Convert back to netcdf
    financing_terms = financing_terms.set_index(["latitude", "longitude"])
    processed_data = financing_terms.to_xarray()
    
    return financing_terms, processed_data    

def calculate_required_waccs(wacc_model, data, lcoe=None):

    # Drop existing data
    data = data.drop_vars('Required_WACC', errors="ignore")

    # Convert LCOE to USD/kWh
    lcoe = lcoe / 1000

    # Extract key figures from the data
    latitudes = data.latitude.values
    longitudes = data.longitude.values
    annual_electricity_production = data['electricity_production'] # kWh for 1 MW
    initial_lcoe = data['Calculated_LCOE'] # USD/KWh for 1 MW

    # Calculate annual costs
    annual_costs = data['Calculated_OPEX'] # USD/kW/year
    capital_costs = data['Calculated_CAPEX'] # USD for 1 MW


    # Get LCOE 
    if lcoe is None:
        lcoe = xr.where(np.isnan(initial_lcoe), np.nan, lcoe)
        data['Benchmark_LCOE'] = lcoe
    else:
        data['Benchmark_LCOE'] = xr.full_like(data['Calculated_CAPEX'], lcoe)

    # Calculate discount factor at each location
    # Ensure that the denominator is not zero or negative
    valid_mask = (annual_electricity_production * lcoe - annual_costs) > 0

    # Apply the calculation only where valid
    discount_factor = xr.where(
        np.isnan(lcoe) | ~valid_mask,
        np.nan,
        capital_costs / ((annual_electricity_production * lcoe) - annual_costs)
    )

    data['Discount_Factor'] = discount_factor

    # Create array of discount factor to WACC values and round discount factor
    discount_rates = np.linspace(0, 0.5, 1001)
    discount_factors_array = wacc_model.calculate_discount_factor_v1(discount_rates)
    xdata = discount_rates
    ydata = discount_factors_array

    # Calculate curve fit
    ylog_data = np.log(ydata)
    curve_fit = np.polyfit(xdata, ylog_data, 2)
    y = np.exp(curve_fit[2]) * np.exp(curve_fit[1]*xdata) * np.exp(curve_fit[0]*xdata**2)


    # Create interpolator
    interpolator = interp1d(ydata, xdata, kind='nearest', bounds_error=False, fill_value=(0.5, 9.99))

    # Use rounded discount factors to calculate WACC values 
    estimated_waccs = interpolator(discount_factor)*100
    estimated_waccs = xr.where(discount_factor < 0, 999, estimated_waccs)
    estimated_waccs = xr.where(discount_factor < 0, np.nan, estimated_waccs)
    wacc_da = xr.DataArray(estimated_waccs, coords={"latitude": latitudes, "longitude":longitudes})
    data['Benchmark_WACC'] = xr.where(np.isnan(initial_lcoe)==True, np.nan, wacc_da)

    return data

def calculate_concessional_needs(wacc_model, concessional_rate, risk_reductions, solar_benchmark, wind_benchmark):
    
    # Calculate the required WACC to reach the benchmark
    solar_benchmark_results = calculate_required_waccs(wacc_model, wacc_model.solar_results, lcoe = solar_benchmark)
    wind_benchmark_results = calculate_required_waccs(wacc_model, wacc_model.wind_results, lcoe = wind_benchmark)

    # Extract WACC
    solar_benchmark_wacc = solar_benchmark_results['Benchmark_WACC']
    wind_benchmark_wacc = wind_benchmark_results['Benchmark_WACC']
    
    # Calculate the fall in WACC required
    wacc_concessional_solar = xr.where(solar_benchmark_results['Calculated_LCOE'] < solar_benchmark, 111, solar_benchmark_wacc)
    wacc_concessional_wind = xr.where(wind_benchmark_results['Calculated_LCOE'] < wind_benchmark, 111, wind_benchmark_wacc)
    
    # Call the concessional finance function to calculate concessionality
    concessional_calcs, processed_data_solar = get_concessionality_v3(wacc_model, wacc_concessional_solar, concessional_rate, "Solar", risk_reductions)
    concessional_calcs, processed_data_wind = get_concessionality_v3(wacc_model, wacc_concessional_wind, concessional_rate, "Wind", risk_reductions)
    
    return processed_data_solar, processed_data_wind

processed_data_solar, processed_data_wind = calculate_concessional_needs(wacc_model, concessional_rate=1, risk_reductions=30, solar_benchmark=60, wind_benchmark=49.5)
wacc_model.plot_data_shading(processed_data_solar['Concessional_Debt_Share'], processed_data_solar.latitude, processed_data_solar.longitude, special_value = 999, hatch_label = "Unable to reach\nunder US$60/MWh", special_value_2 = 111, hatch_label_2="Already below\nUS$60/MWh\nat current WACCs", tick_values = [0, 25, 50, 75, 100], cmap="YlOrRd", title="Required\nShare Of\nConcessional\nFinancing (%)\n", graphmarking="a", extend_set="neither", filename="Solar_Concessional_30_RR")
wacc_model.plot_data_shading(processed_data_wind['Concessional_Debt_Share'], processed_data_wind.latitude, processed_data_wind.longitude, special_value = 999, hatch_label = "Unable to reach\nunder US$49.5/MWh", special_value_2 = 111, hatch_label_2="Already below\nUS$49.5/MWh\nat current WACCs", tick_values = [0, 25, 50, 75, 100], cmap="YlGnBu", title="Required\nShare Of\nConcessional\nFinancing (%)\n", graphmarking="b", extend_set="neither", filename="Wind_Concessional_30_RR")


processed_data_solar, processed_data_wind = calculate_concessional_needs(wacc_model, concessional_rate=1, risk_reductions=0, solar_benchmark=60, wind_benchmark=49.5)
wacc_model.plot_data_shading(processed_data_solar['Concessional_Debt_Share'], processed_data_solar.latitude, processed_data_solar.longitude, special_value = 999, hatch_label = "Unable to reach\nunder US$60/MWh", special_value_2 = 111, hatch_label_2="Already below\nUS$60/MWh\nat current WACCs", tick_values = [0, 25, 50, 75, 100], cmap="YlOrRd", title="Required\nShare Of\nConcessional\nFinancing (%)\n", graphmarking="a", extend_set="neither", filename="Solar_Concessional_0_RR")
wacc_model.plot_data_shading(processed_data_wind['Concessional_Debt_Share'], processed_data_wind.latitude, processed_data_wind.longitude, special_value = 999, hatch_label = "Unable to reach\nunder US$49.5/MWh", special_value_2 = 111, hatch_label_2="Already below\nUS$49.5/MWh\nat current WACCs", tick_values = [0, 25, 50, 75, 100], cmap="YlGnBu", title="Required\nShare Of\nConcessional\nFinancing (%)\n", graphmarking="b", extend_set="neither", filename="Wind_Concessional_0_RR") 
                            
                            def plot_supply_curves(wacc_model, results, country_index):
    
    # Select country
    national_results = xr.where(wacc_model.land_grids == country_index, results, np.nan)
    
    # Convert into dataframe
    national_df = national_results.to_dataframe()
    
    # Create axis
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
    
    # Color 
    color = ["red", "blue", "black"]
    
    # Sort each of the three variables
    for i, variable in enumerate(["Calculated_LCOE", "Uniform_LCOE", "Subnational_LCOE"]):
        
        # Extract data
        extracted_data = national_df[[variable, "electricity_production"]]
        
        # Sort values
        extracted_data = extracted_data.sort_values(by=[variable], ascending=True)
        
        # Create cumulative sum
        extracted_data['cumulative_production'] = extracted_data['electricity_production'].cumsum()
        
        # Plot
        ax.plot(extracted_data['cumulative_production'], extracted_data[variable], color=color[i], lw=2.5, label=variable, linestyle="--")
        
    # Set axis limits
    ax.set_ylim([0, 100])
    ax.text(0.02, 0.9, country_index, transform=ax.transAxes, fontsize=15, fontweight='bold')
        
solar_results = wacc_model.solar_results
for i in np.arange(1, 251, 1)
                            
                            
                            
                            
                            
                            
                            
 def run_TIAM_regions(wacc_model):
        
    # Extract the relevant data
    TIAM_regions = wacc_model.TIAM_regions 
    
    # Specify input
    solar_results = wacc_model.solar_lcoe
    onshore_results = wacc_model.wind_lcoe
    offshore_results = wacc_model.offshore_lcoe
    
    # Specify uniform factors
    solar_uf = wacc_model.solar_pv_uf
    onshore_uf = wacc_model.onshore_uf
    offshore_uf = wacc_model.offshore_uf
    
    # Specify postprocessor
    postprocessor = wacc_model.postprocessor

    # Call postprocessing function to create the corresponding supply curve
    plot_TIAM_lcoe(postprocessor, solar_results, onshore_results, offshore_results, solar_uf, onshore_uf, offshore_uf)

        
def plot_TIAM_lcoe(postprocessor, solar_filtered_results, onshore_filtered_results, offshore_filtered_results, solar_uf, onshore_uf, offshore_uf):
    
    # Get Solar and Wind Datasets with technical potential
    solar_ds = postprocessor.get_supply_curves_v2(solar_filtered_results,  "Solar")
    wind_ds = postprocessor.get_supply_curves_v2(onshore_filtered_results, "Onshore Wind")
    offshore_ds = postprocessor.get_supply_curves_v2(offshore_filtered_results, "Offshore Wind")
    
    
    # Plot corresponding results separately
    produce_lcoe_potential_v1(postprocessor, solar_ds, xlim=250000, graphmarking="a", uniform_value=solar_uf, position="upper", filename="LCOE_Supply_Solar", technology="Solar")
    produce_lcoe_potential_v1(postprocessor, wind_ds, xlim=250000, graphmarking="b", uniform_value=onshore_uf, position="upper", filename="LCOE_Supply_Onshore", technology="Onshore Wind")
    produce_lcoe_potential_v1(postprocessor, offshore_ds, xlim=2500, technology="Offshore Wind", graphmarking="c", uniform_value=offshore_uf, position="lower", filename="LCOE_Supply_Offshore")


def produce_lcoe_potential_v1(postprocessor, supply_ds, filename=None, graphmarking=None, position=None, uniform_value=None, technology=None, region_code=None, subnational=None, xlim=None):
    
    def thousands_format(x, pos):
        return f'{int(x):,}'
    
    # Convert the dataset into a dataframe
    supply_df = supply_ds.to_dataframe()
    
    # Merge with the country and region mapping
    merged_supply_df = pd.merge(supply_df, postprocessor.country_mapping.rename(columns={"index":"Country"}), how="left", on="Country")

    # Merge with country_mapping to give
    supply_df = merged_supply_df.copy().dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")

    # Create two copies
    uniform_df = merged_supply_df.copy().dropna(axis=0, subset=["Uniform_LCOE", "Country"], how="all")
    wacc_df = merged_supply_df.copy().dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")
    if subnational is not None:
        subnational_df = merged_supply_df.copy().dropna(axis=0, subset=["Subnational_LCOE", "Country"], how="all")

    # Convert units to TWh
    supply_df['technical_potential'] = supply_df['technical_potential'] / 1e+09
    uniform_df['technical_potential'] = uniform_df['technical_potential'] / 1e+09
    wacc_df['technical_potential'] = wacc_df['technical_potential'] / 1e+09
    if subnational is not None:
        subnational_df['technical_potential'] = subnational_df['technical_potential'] / 1e+09

    # For the WACC case, sort values and calculate cumulative sum
    wacc_sorted_df = wacc_df.sort_values(by=['Region', 'Calculated_LCOE'], ascending=True)
    wacc_sorted_df['cumulative_potential'] = wacc_sorted_df.groupby('Region')['technical_potential'].cumsum()
    wacc_grouped = wacc_sorted_df.groupby('Region')

    # For the Uniform WACC case, sort values and calculate cumulative sum
    uniform_sorted_df = uniform_df.sort_values(by=['Region', 'Uniform_LCOE'], ascending=True)
    uniform_sorted_df['cumulative_potential'] = uniform_sorted_df.groupby('Region')['technical_potential'].cumsum()
    uniform_grouped = uniform_sorted_df.groupby('Region')
    
    if subnational is not None:
        # For the Uniform WACC case, sort values and calculate cumulative sum
        subnational_sorted_df = subnational_df.sort_values(by=['Region', 'Uniform_LCOE'], ascending=True)
        subnational_sorted_df['cumulative_potential'] = subnational_sorted_df.groupby('Region')['technical_potential'].cumsum()
        subnational_grouped = subnational_sorted_df.groupby('Region')

    # Set regional colour scheme
    region_colors = {
    "AFR": "purple",          # Stays the same, distinct
    "AUS": "dodgerblue",      # Changed from "cornflowerblue" to a more vivid blue
    "FSU": "gold",            # Changed from "yellow" to "gold" for richer contrast
    "CAN": "lightgray",       # Changed from "silver" to "lightgray" for a softer tone
    "CHN": "red",             # Stays the same, highly distinct
    "CSA": "forestgreen",     # Changed from "green" to "forestgreen" for a deeper tone
    "IND": "darkorange",      # Changed from "orange" to "darkorange" for higher contrast
    "JPN": "teal",            # Changed from "cyan" to "teal" for a stronger, less bright tone
    "MEA": "goldenrod",       # Changed from "olive" to "goldenrod" for a less muted yellow
    "MEX": "black",           # Stays the same, highly distinct
    "ODA": "hotpink",         # Changed from "pink" to "hotpink" for vibrancy
    "EEU": "limegreen",       # Changed from "darkgreen" to "limegreen" for brightness
    "KOR": "chocolate",       # Changed from "sandybrown" to "chocolate" for richer tone
    "USA": "firebrick",       # Changed from "crimson" to "firebrick" for a slightly muted red
    "WEU": "navy"             # Changed from "darkblue" to "navy" for stronger differentiation
}
    # Plot the results
    if technology == "Offshore Wind":
        width_ratio = [1, 1]
    else:
        width_ratio = [2, 1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8),gridspec_kw = {'wspace':0, 'hspace':0, 'width_ratios': width_ratio})
    region_list = ["AFR", "FSU", "CSA", "MEA", "ODA", "EEU"]
    other_regions = list(set(region_colors.keys()) - set(region_list))
    selected_region_colors = dict((region, region_colors[region]) for region in region_list if region in region_colors)
    other_region_colors = dict((region, region_colors[region]) for region in other_regions if region in region_colors)
    
    # Loop over the regions specified
    for region in region_list:

        # Extract color for region
        region_shading = region_colors.get(region, "grey")
    
    
        # Get cumulative production data for the region
        country_wacc_potential = wacc_grouped[['cumulative_potential']].get_group(region)
        uniform_wacc_potential = uniform_grouped[['cumulative_potential']].get_group(region)
        
        # Get corresponding lcoe
        country_wacc_lcoe = wacc_grouped[['Calculated_LCOE']].get_group(region)
        uniform_wacc_lcoe = uniform_grouped[['Uniform_LCOE']].get_group(region)

        # Produce plots of LCOE against supply for each region
        ax1.plot(country_wacc_potential,country_wacc_lcoe , color=region_shading, linestyle="-")
        ax1.plot(uniform_wacc_potential,uniform_wacc_lcoe, color=region_shading, linestyle="--")
        
        # Plot the subnational if applicable
        if subnational is not None:
            subnational_wacc_potential = subnational_grouped[['cumulative_potential']].get_group(region)
            subnational_wacc_lcoe = subnational_grouped[['Subnational_LCOE']].get_group(region)
            ax1.plot(subnational_wacc_potential,subnational_wacc_lcoe, color=region_shading, linestyle=":")
            
    # Create the legend
    handles = [plt.Line2D([0], [0], color=color, lw=4, label=region) for region, color in region_colors.items()]
    ax1.legend(handles=handles, title="Regions", loc=position+ " right", ncol=5, fontsize=12, title_fontsize=15)
    
    # Set labels
    ax1.set_ylim(0, 250)
    ax1.set_xlim(0, xlim)
    ax1.set_ylabel('Levelised Cost of Electricity\n(USD/MWh, '+ technology + ')', fontsize=18)
    ax1.set_xlabel('Developing Regions\nAnnual Electricity Potential (TWh/year)', fontsize=18)
    ax1.xaxis.set_major_formatter(FuncFormatter(thousands_format))

    # Set the size of x and y-axis tick labels
    ax1.tick_params(axis='x', labelsize=15)  # Adjust the labelsize as needed
    ax1.tick_params(axis='y', labelsize=15)  # Adjust the labelsize as needed
            
    # Loop over the regions specified
    for region in other_regions:

        # Extract color for region
        region_shading = region_colors.get(region, "grey")
    
    
        # Get cumulative production data for the region
        country_wacc_potential = wacc_grouped[['cumulative_potential']].get_group(region)
        uniform_wacc_potential = uniform_grouped[['cumulative_potential']].get_group(region)
        
        # Get corresponding lcoe
        country_wacc_lcoe = wacc_grouped[['Calculated_LCOE']].get_group(region)
        uniform_wacc_lcoe = uniform_grouped[['Uniform_LCOE']].get_group(region)

        # Produce plots of LCOE against supply for each region
        ax2.plot(country_wacc_potential,country_wacc_lcoe , color=region_shading, linestyle="-")
        ax2.plot(uniform_wacc_potential,uniform_wacc_lcoe, color=region_shading, linestyle="--")
        
        # Plot the subnational if applicable
        if subnational is not None:
            subnational_wacc_potential = subnational_grouped[['cumulative_potential']].get_group(region)
            subnational_wacc_lcoe = subnational_grouped[['Subnational_LCOE']].get_group(region)
            ax2.plot(subnational_wacc_potential,subnational_wacc_lcoe, color=region_shading, linestyle=":")
            
    # Create the legend
    solid_line = plt.Line2D([0], [0], color="black", lw=4, linestyle='-', label='Estimated country- and\ntechnology- WACCs')
    dashed_line = plt.Line2D([0], [0], color="black", lw=4, linestyle='--', label=f'Uniform {uniform_value:0.1f}% WACC')
    style_handles = [solid_line, dashed_line]
    ax2.legend(handles=style_handles, title="Cost of Capital", loc=position+ " left", ncol=1, fontsize=12, title_fontsize=15)
    
    # Set labels
    ax2.set_ylim(0, 250)
    if technology == "Offshore Wind":
        ax2.set_xlim(0, xlim)
    else:
        ax2.set_xlim(0, xlim/2)

    ax2.set_xlabel('Developed & Industrialising Regions\nAnnual Electricity Potential (TWh/year)', fontsize=18)
    ax2.xaxis.set_major_formatter(FuncFormatter(thousands_format))

    # Set the size of x and y-axis tick labels
    ax2.tick_params(axis='x', labelsize=15)  # Adjust the labelsize as needed
    ax2.tick_params(axis='y', labelsize=0)  # Adjust the labelsize as needed

    if xlim is not None:
        if technology == "Offshore Wind":
            ax1.xaxis.set_ticks(np.arange(0, xlim, 500))
            ax2.xaxis.set_ticks(np.arange(0, (xlim), 500))
        else:
            ax1.xaxis.set_ticks(np.arange(0, xlim, 25000))
            ax2.xaxis.set_ticks(np.arange(0, (xlim/2), 25000))

    if graphmarking is not None:
        ax1.text(0.02, 0.94, graphmarking, transform=ax1.transAxes, fontsize=20, fontweight='bold')

    if region_code is not None:
        ax1.text(0.15, 0.9, technology + "\n" + region_code, transform=ax1.transAxes, fontsize=20, fontweight='bold', ha="center", va="center")

    if filename is not None:
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()



run_TIAM_regions(wacc_model)


def get_locational_capacity(postprocessor, tech_potential_ds, GW_target, country_deployment_limits, technology):
    
    # Convert into a dataframe
    tech_potential_df = tech_potential_ds.to_dataframe()
     
    # Add in region and country to the dataframe
    potential_df_unindexed = pd.merge(tech_potential_df.reset_index(), postprocessor.country_mapping.rename(columns={"index":"Country"})[['Country', 'Region']], how="left", on="Country")
    potential_df = potential_df_unindexed.set_index(["latitude", "longitude"])
    print(potential_df)
    potential_df  = potential_df[~potential_df .index.duplicated()]
    
    # Order the dataframe by Uniform LCOE
    uniform_ordered_df = potential_df.copy().sort_values(by=["Uniform_LCOE"], ascending=True)
    
    # Order the dataframe by Calculated LCOE
    national_ordered_df = potential_df.copy().sort_values(by=["Calculated_LCOE"], ascending=True)
    
    # Process deployment limits based on solar and wind targets
    if technology == "Solar":
        country_deployment_limits['national_deployment_limit'] = 2 * country_deployment_limits['Solar_Target']
    else:
        country_deployment_limits['national_deployment_limit'] = 2 * country_deployment_limits['Wind_Target']
        
    # For countries without targets, set the limit as 50% of current installed capacity
    #country_deployment_limits['national_deployment_limit'] = np.where(np.isnan(country_deployment_limits['national_deployment_limit']), 0.5 * country_deployment_limits['Total'], country_deployment_limits['national_deployment_limit'])
    
    # For countries without data on installed capacity, set the limit as 10 GW
    country_deployment_limits['national_deployment_limit'] = np.where(np.isnan(country_deployment_limits['national_deployment_limit']), 100,country_deployment_limits['national_deployment_limit'])
    
    
    # Calculate with national limits
    uniform_ordered_df = tripling_renewables_locations(uniform_ordered_df.reset_index(), GW_target, country_deployment_limits)
    national_ordered_df = tripling_renewables_locations(national_ordered_df.reset_index(), GW_target, country_deployment_limits)
    
    # Set indexes
    uniform_df = uniform_ordered_df.set_index(["latitude", "longitude"])
    national_df = national_ordered_df.set_index(["latitude", "longitude"])
    
    # Remove duplicated indexes 
    uniform_df_reindexed = uniform_df[~uniform_df.index.duplicated()]
    national_df_reindexed = national_df[~national_df.index.duplicated()]
    
    # Convert to xarray
    uniform_ds = uniform_df_reindexed.to_xarray()
    national_ds = national_df_reindexed.to_xarray()
    
    return national_ds, uniform_ds
 
    
def tripling_renewables_locations(potential_df, GW_target, country_deployment_limits):


    # Group by country 
    country_grouped = potential_df.groupby('Country')
    
    # Extract cumulative capacity potential 
    potential_df['national_cumulative_capacity'] = potential_df.groupby('Country')['capacity_GW'].cumsum()
    
    # Read in national deployment limit
    storage_df = pd.merge(potential_df, country_deployment_limits, how="left", on="Country")
    
    # Apply limits to each country
    storage_df['national_cumulative_capacity'] = np.where(storage_df['national_cumulative_capacity'] > storage_df['national_deployment_limit'], np.nan, storage_df['national_cumulative_capacity'])
    
    # Recalculate cumulative capacity
    storage_df['national_capacity_GW'] = np.where(np.isnan(storage_df['national_cumulative_capacity']), np.nan, storage_df['capacity_GW'])
    
    # Extract new cumulative sum of capacity 
    storage_df['cumulative_national_GW'] = storage_df['national_capacity_GW'].cumsum()
    
    # Apply flag
    storage_df['GW_national_flag'] = np.where(storage_df['cumulative_national_GW']< GW_target, 1, np.nan)
    storage_df['GW_national_flag'] = np.where(np.isnan(storage_df['national_capacity_GW']), np.nan, storage_df['GW_national_flag'])
    
    return storage_df
                                         
                                         
                                  
                                
    
country_deployment_limits = pd.read_csv("country_deployment_limits.csv")
solar_ds = wacc_model.postprocessor.get_supply_curves_v2(wacc_model.solar_lcoe,  "Solar")
wind_ds = wacc_model.postprocessor.get_supply_curves_v2(wacc_model.wind_lcoe, "Onshore Wind")
national_solar, uniform_solar = get_locational_capacity(wacc_model.postprocessor, solar_ds, 11000, country_deployment_limits, "Solar")
national_onshore, uniform_onshore = get_locational_capacity(wacc_model.postprocessor, wind_ds, 11000, country_deployment_limits, "Wind")

wacc_model.postprocessor.plot_data_shading(national_solar['GW_national_flag'], national_solar.latitude, national_solar.longitude, tick_values=[0, 1], cmap="YlOrRd") 
wacc_model.postprocessor.plot_data_shading(uniform_solar['GW_national_flag'], national_solar.latitude, national_solar.longitude, tick_values=[0, 1], cmap="YlOrRd") 
wacc_model.postprocessor.plot_data_shading(national_onshore['GW_national_flag'], national_solar.latitude, national_solar.longitude, tick_values=[0, 1], cmap="YlGnBu") 
wacc_model.postprocessor.plot_data_shading(uniform_onshore['GW_national_flag'], national_solar.latitude, national_solar.longitude, tick_values=[0, 1], cmap="YlGnBu")         
                            
def get_supply_curves_v3(postprocessor, data, technology, offshore=None, LCOE_cutoff=None):

    # Extract required parameters
    annual_production = data['electricity_production']
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values

    # Get area and utilisations
    grid_areas = postprocessor.get_areas(annual_production)
    utilisation_factors = postprocessor.get_utilisations(annual_production, technology)

    # Set out constants
    if technology == "Onshore Wind":
        power_density = 6520 # kW/km2
        cutoff = 0.18
    elif technology == "Offshore Wind":
        power_density = 4000 # kW/km2
        cutoff = 0.18
    elif technology == "Solar":
        power_density = 32950  # kW/km2
        cutoff = 0.1
    installed_capacity = 1000

    # Apply cut off factors
    utilisation_factors = xr.where(data['CF']<cutoff, 0, utilisation_factors)
    if LCOE_cutoff is not None:
        data['Calculated_LCOE'] = xr.where(data['CF']<cutoff, np.nan, data['Calculated_LCOE'])
        data['Uniform_LCOE'] = xr.where(data['CF']<cutoff, np.nan, data['Uniform_LCOE'])

    # Scale annual electricity production by power density
    max_installed_capacity = power_density * grid_areas['area'] * utilisation_factors
    ratios = max_installed_capacity / installed_capacity
    technical_potential = annual_production * ratios

    # Include additional data into the dataset
    data['technical_potential'] = technical_potential
    data['capacity_GW'] = max_installed_capacity / 1e+06 # convert from kW to GW 
    if technology == "Offshore Wind":
        data['Country'] = postprocessor.country_grids['sea']
    else:
        data['Country'] = postprocessor.country_grids['land']

    return data



def get_locational_capacity_v2(postprocessor, tech_potential_ds, GW_target, country_deployment_limits, technology, regional_deployment_limits=None):
    
    # Convert into a dataframe
    tech_potential_df = tech_potential_ds.to_dataframe()
    
    # Apply restriction
    scaling_factor = 0.05
    tech_potential_df['capacity_GW'] = tech_potential_df['capacity_GW'] * scaling_factor
    tech_potential_df['technical_potential'] = tech_potential_df['technical_potential'] * scaling_factor
    
     
    # Add in region and country to the dataframe
    potential_df_unindexed = pd.merge(tech_potential_df.reset_index(), postprocessor.country_mapping.rename(columns={"index":"Country"})[['Country', 'Region']], how="left", on="Country")
    potential_df = potential_df_unindexed.set_index(["latitude", "longitude"])
    potential_df_reindexed  = potential_df[~potential_df .index.duplicated()]
    
    # Order the dataframe by Uniform LCOE
    uniform_ordered_df = potential_df_reindexed.copy().sort_values(by=["Uniform_LCOE"], ascending=True)
    
    # Order the dataframe by Calculated LCOE
    national_ordered_df = potential_df_reindexed.copy().sort_values(by=["Calculated_LCOE"], ascending=True)
    
    # Process deployment limits based on solar and wind targets
    if technology == "Solar":
        regional_deployment_limits['regional_deployment_limit'] = 1.25 * regional_deployment_limits['Solar_Target'] 
    else:
        regional_deployment_limits['regional_deployment_limit'] = 1.25 * regional_deployment_limits['Wind_Target']

    regional_deployment_limits['regional_deployment_limit'] = regional_deployment_limits['regional_deployment_limit'].fillna(value=100)
    
    # Calculate with national limits
    uniform_limited_df = tripling_renewables_regional(uniform_ordered_df.reset_index(), GW_target, regional_deployment_limits)
    national_limited_df = tripling_renewables_regional(national_ordered_df.reset_index(), GW_target, regional_deployment_limits)
    
    # Set indexes
    uniform_df = uniform_limited_df.set_index(["latitude", "longitude"])
    national_df = national_limited_df.set_index(["latitude", "longitude"])
    
    # Remove duplicated indexes 
    uniform_df_reindexed = uniform_df[~uniform_df.index.duplicated()]
    national_df_reindexed = national_df[~national_df.index.duplicated()]
    
    # Convert to xarray
    uniform_ds = uniform_df_reindexed.to_xarray()
    national_ds = national_df_reindexed.to_xarray()
    
    return national_ds, uniform_ds
 
def tripling_renewables_regional(potential_df, GW_target, regional_deployment_limits):
    
    # Extract cumulative capacity potential 
    potential_df['regional_cumulative_capacity'] = potential_df.groupby('Region')['capacity_GW'].cumsum()
    storage_df = pd.merge(potential_df, regional_deployment_limits, how="left", on="Region")
        
    # Name 
    processing_df = storage_df.copy()
    
    # Apply limits to each country
    processing_df['national_regional_capacity'] = np.where(processing_df['regional_cumulative_capacity'] > processing_df['regional_deployment_limit'], np.nan, processing_df['regional_cumulative_capacity'])
    
    # Recalculate cumulative capacity
    processing_df['regional_capacity_GW'] = np.where(np.isnan(processing_df['regional_cumulative_capacity']), np.nan, processing_df['capacity_GW'])

    # Extract new cumulative sum of capacity 
    processing_df['cumulative_regional_GW'] = processing_df['regional_capacity_GW'].cumsum()

    # Apply flag
    processing_df['GW_national_flag'] = np.where(processing_df['cumulative_regional_GW']< GW_target, 1, np.nan)
    processing_df['GW_national_flag'] = np.where(np.isnan(processing_df['regional_capacity_GW']), np.nan, processing_df['GW_national_flag'])

    return processing_df
    


def combine_wind_solar(postprocessor, country_deployment_limits, regional_deployment_limits):
    
    def calculate_output(dataset, scenario):
        
        # Get wind
        wind_output = xr.where(dataset['COP28_Wind'] == 1, dataset['wind_potential'], np.nan)
        wind_total = wind_output.sum(skipna=True)/1e+09
        wind_cf = xr.where(dataset['COP28_Wind'] == 1, dataset['Wind_CF'], np.nan)
        wind_mean_cf = wind_cf.mean(skipna=True)*100
        
        # Get solar output
        solar_output = xr.where(dataset['COP28_Solar'] == 1, dataset['solar_potential'], np.nan)
        solar_cf = xr.where(dataset['COP28_Solar'] == 1, dataset['Solar_CF'], np.nan)
        solar_total = solar_output.sum(skipna=True)/1e+09
        solar_mean_cf = solar_cf.mean(skipna=True)*100
        
        # Print output
        print(f"Output for {scenario} is {wind_total:0.2f}TWh for onshore wind and {solar_total:0.2f}TWh for solar PV")
        print(f"Mean Capacity Factor for {scenario} is {wind_mean_cf:0.2f}% for onshore wind and {solar_mean_cf:0.2f}% for solar PV")
        
        return wind_total, solar_total

    # Get onshore wind and solar results
    solar_ds = get_supply_curves_v3(wacc_model.postprocessor, wacc_model.solar_lcoe,  "Solar")
    wind_ds = get_supply_curves_v3(wacc_model.postprocessor, wacc_model.wind_lcoe, "Onshore Wind")

    # Get solar results
    national_solar, uniform_solar = get_locational_capacity_v2(wacc_model.postprocessor, solar_ds, 11000, country_deployment_limits, "Solar", regional_deployment_limits=regional_deployment_limits)

    # Get wind results
    national_onshore, uniform_onshore = get_locational_capacity_v2(wacc_model.postprocessor, wind_ds, 11000, country_deployment_limits, "Wind",regional_deployment_limits=regional_deployment_limits)

    # Combine Uniform scenario
    uniform_solar = uniform_solar.rename(name_dict={"GW_national_flag":"COP28_Solar", "technical_potential":"solar_potential", "CF":"Solar_CF", "capacity_GW":"solar_GW"})
    uniform_onshore = uniform_onshore.rename(name_dict={"GW_national_flag":"COP28_Wind", "technical_potential":"wind_potential", "CF":"Wind_CF", "capacity_GW":"wind_GW"})
    uniform_results = xr.merge([uniform_solar[['Country', 'COP28_Solar', 'solar_potential', "Solar_CF", "solar_GW"]], uniform_onshore[['COP28_Wind', 'wind_potential', "Wind_CF", "wind_GW"]]], join="left")

    # Combine National scenario
    national_solar = national_solar.rename(name_dict={"GW_national_flag":"COP28_Solar", "technical_potential":"solar_potential", "CF":"Solar_CF", "capacity_GW":"solar_GW"})
    national_onshore = national_onshore.rename(name_dict={"GW_national_flag":"COP28_Wind", "technical_potential":"wind_potential", "CF":"Wind_CF","capacity_GW":"wind_GW"})
    national_results = xr.merge([national_solar[['Country', 'COP28_Solar', 'solar_potential', "Solar_CF", "solar_GW"]], national_onshore[['COP28_Wind', 'wind_potential', "Wind_CF", "wind_GW"]]], join="left")
    
    # Sum up technical potential
    wind_uniform, solar_uniform = calculate_output(uniform_results, "Uniform")
    wind_national, solar_national = calculate_output(national_results, "National")
    print(f"Wind increases by {wind_uniform/wind_national}x")
    print(f"Solar increases by {solar_uniform/solar_national}x")

    return uniform_results, national_results
                                         
def plot_renewables_distribution(dataset, filename=None, graphmarking=None, scenario=None):      
    
    # Get latitudes and longitudes
    latitudes = dataset.latitude.values
    longitudes = dataset.longitude.values

    # Get wind and solar data
    wind_data = dataset['COP28_Wind']
    solar_data = dataset['COP28_Solar']
    wind_color = "navy"
    solar_color = "orange"

    # Create a figure and axes objects
    fig = plt.figure(figsize=(30, 15), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Apply shading for wind and solar
    wind_overlay = np.where(wind_data == 1, 1, np.nan)
    solar_overlay = np.where(solar_data == 1, 1, np.nan)
    wind_hatching = ax.contourf(longitudes, latitudes, wind_overlay, colors=wind_color, linewidth=0.05, transform=ccrs.PlateCarree(), alpha=0.8)
    solar_hatching = ax.contourf(longitudes, latitudes, solar_overlay, colors=solar_color, linewidth=0.05, transform=ccrs.PlateCarree(), alpha=0.8)

    # set the extent and aspect ratio of the plot
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
    aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
    ax.set_aspect(1)

    # add axis labels and a title
    ax.set_xlabel('Longitude', fontsize=30)
    ax.set_ylabel('Latitude', fontsize=30)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
    ax.add_feature(borders, edgecolor='gray', linestyle=':')
    ax.coastlines()
    ax.coastlines()

    # Add Legend
    hatch_patches=[]
    wind_patch = Patch(facecolor=wind_color, edgecolor='black', hatch="", label="Onshore Wind")
    solar_patch = Patch(facecolor=solar_color, edgecolor='black', hatch="", label="Solar PV")
    hatch_patches.append(solar_patch)
    hatch_patches.append(wind_patch)
    ax.legend(title=scenario + " Scenario:\nLowest cost locations\nfor 11,000 GW", handles=hatch_patches, loc='lower left', fontsize=20, title_fontsize=25, alignment="left") 

    if graphmarking is not None:
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

    if filename is not None:
        plt.savefig(filename + ".png", bbox_inches="tight")

    return

                                    
country_deployment_limits = pd.read_csv("country_deployment_limits.csv")
regional_deployment_limits = pd.read_csv("regional_deployment_limits.csv")

uniform_results, national_results = combine_wind_solar(wacc_model.postprocessor, country_deployment_limits, regional_deployment_limits)
plot_renewables_distribution(uniform_results, scenario="Uniform", filename="Uniform_Tripling", graphmarking="b")
plot_renewables_distribution(national_results, scenario="National", filename="National_Tripling", graphmarking="a")     

def calculate_abatement_costs(CI_data, capacity_data):
    
    # Perform the calculation
    solar_tCO2 = CI_data /1000000 * capacity_data['solar_potential'] * 20 * 0.5
    wind_tCO2 = CI_data /1000000 * capacity_data['wind_potential'] * 20 * 0.5
    capacity_data['solar_abatement'] = xr.where(capacity_data['COP28_Solar'] == 1, solar_tCO2, np.nan)
    capacity_data['wind_abatement'] = xr.where(capacity_data['COP28_Wind'] == 1, wind_tCO2, np.nan)
    
    return capacity_data

uniform_results = calculate_abatement_costs(country_CI, uniform_results)
national_results = calculate_abatement_costs(country_CI, national_results)

wacc_model.postprocessor.plot_data_shading(national_results['wind_abatement']/1000000, national_results.latitude.values, national_results.longitude.values,tick_values=[0, 2, 4, 6, 8, 10, 15, 20], cmap="Purples")
wacc_model.postprocessor.plot_data_shading(uniform_results['wind_abatement']/1000000, national_results.latitude.values, national_results.longitude.values,tick_values=[0, 2, 4, 6, 8, 10, 15, 20], cmap="Purples")

def calculate_abatement_costs(CI_data, capacity_data):
    
    # Perform the calculation
    solar_tCO2 = CI_data /1000000 * capacity_data['solar_potential'] * 20 * 0.5
    wind_tCO2 = CI_data /1000000 * capacity_data['wind_potential'] * 20 * 0.5
    capacity_data['solar_abatement'] = xr.where(capacity_data['COP28_Solar'] == 1, solar_tCO2, np.nan)
    capacity_data['wind_abatement'] = xr.where(capacity_data['COP28_Wind'] == 1, wind_tCO2, np.nan)
    
    return capacity_data

uniform_results = calculate_abatement_costs(country_CI, uniform_results)
national_results = calculate_abatement_costs(country_CI, national_results)
print(f"For Solar, the Uniform abatement is {np.nansum(uniform_results['solar_abatement'])} and the National is {np.nansum(national_results['solar_abatement'])}")
print(f"For Wind, the Uniform abatement is {np.nansum(uniform_results['wind_abatement'])} and the National is {np.nansum(national_results['wind_abatement'])}")
      
      
# Call abatement function
wacc_model.postprocessor.plot_data_shading(national_results['wind_abatement']/1000000, national_results.latitude.values, national_results.longitude.values,tick_values=[0, 2, 4, 6, 8, 10, 15, 20], cmap="Purples", title="Wind-National:\nAbatement\nPotential\n(MtCO2)\n", graphmarking="a", filename="COP_Wind_Abatement_National")
wacc_model.postprocessor.plot_data_shading(uniform_results['wind_abatement']/1000000, national_results.latitude.values, national_results.longitude.values,tick_values=[0, 2, 4, 6, 8, 10, 15, 20], cmap="Purples", title="Wind-Uniform:\nAbatement\nPotential\n(MtCO2)\n", graphmarking="b", filename="COP_Wind_Abatement_Uniform")
wacc_model.postprocessor.plot_data_shading(national_results['solar_abatement']/1000000, national_results.latitude.values, national_results.longitude.values,tick_values=[0, 2, 4, 6, 8, 10, 15, 20], cmap="Purples", title="Solar-National:\nAbatement\nPotential\n(MtCO2)\n", graphmarking="a", filename="COP_Solar_Abatement_National")
wacc_model.postprocessor.plot_data_shading(uniform_results['solar_abatement']/1000000, national_results.latitude.values, national_results.longitude.values,tick_values=[0, 2, 4, 6, 8, 10, 15, 20], cmap="Purples", title="Solar-Uniform:\nAbatement\nPotential\n(MtCO2)\n", graphmarking="b", filename="COP_Solar_Abatement_Uniform")

def plot_lcoe_abatement(wacc_model, lcoe_data, uniform_data, national_data, technology):
    
    # Specify location to extract
    if technology == "Solar":
        abatement = "solar_abatement"
    else:
        abatement = "wind_abatement"
        
    # Extract abatement
    uniform_abatement = uniform_data[abatement]
    national_abatement = uniform_data[abatement]
    
    # Extract LCOE and abatement
    uniform_lcoe = xr.where(np.isnan(uniform_abatement), np.nan, lcoe_data['Uniform_LCOE'])
    calculated_lcoe = xr.where(np.isnan(national_abatement), np.nan, lcoe_data['Calculated_LCOE'])
    
    # Combine abatement for Uniform
    lcoe_data['abatement'] = uniform_data[abatement]
    lcoe_data['Uniform_LCOE'] = uniform_lcoe
    
    # Combine abatement for National
    lcoe_data['abatement'] = national_data[abatement]
    lcoe_data['Calculated_LCOE'] = calculated_lcoe
    lcoe_data['Country'] = uniform_data['Country']
    
    # Merge with the region mapping
    lcoe_data = lcoe_data.to_dataframe()
    merged_data = pd.merge(lcoe_data, wacc_model.postprocessor.country_mapping.rename(columns={"index":"Country"})[['Country', 'Region']], how="left", on="Country")
    
    return merged_data
    
    

    
merged_data = plot_lcoe_abatement(wacc_model, wacc_model.solar_lcoe, uniform_results, national_results, "Solar")
                                
                            
                            
                            