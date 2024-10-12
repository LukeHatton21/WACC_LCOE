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