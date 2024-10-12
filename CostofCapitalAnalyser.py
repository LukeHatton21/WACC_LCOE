import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
from sklearn import linear_model
import scipy.stats as stats



class CoCModel:
    def __init__(self,running_folder, generation_data, crp_data, irena_waccs, aures_waccs, country_codes, aures_diacore, OECD_IR, IMF_IR, GDP, GDP_Change, collated_IR, electricity_prices, tax_data, irena_2022, rise_data, country_mapping):
        # Read in the input data - generation and CRPs
        self.generation_data = pd.read_csv(running_folder + generation_data)
        self.crp_data = pd.read_csv(running_folder + crp_data)
        self.irena_waccs = pd.read_csv(running_folder + irena_waccs)
        self.aures_waccs = pd.read_csv(running_folder + aures_waccs)
        self.aures_diacore = pd.read_csv(running_folder + aures_diacore)
        self.country_codes = pd.read_csv(running_folder + country_codes)
        self.electricity_prices = pd.read_csv(running_folder + electricity_prices)
        self.tax_data = pd.read_csv(running_folder + tax_data)
        self.electricity_prices = self.electricity_prices[["Country Name", "Country code", "Electricity_Price"]]
        
        # Read in Interest Rate Data from OECD and IMF
        self.OECD_IR = pd.read_csv(running_folder + OECD_IR)  
        self.IMF_IR = pd.read_csv(running_folder + IMF_IR) 
        self.collated_IR = pd.read_csv(running_folder + collated_IR)
        self.OECD_IR.columns = self.OECD_IR.columns.map(str)
        self.IMF_IR.columns = self.IMF_IR.columns.map(str)
        self.collated_IR.columns = self.collated_IR.columns.map(str)
        
        # Read in IRENA Cost of Debt and Equity
        self.irena_disaggregated = pd.read_csv(running_folder + irena_2022)
        self.irena_disaggregated.columns = self.irena_disaggregated.columns.map(str)
        
        # Read in RISE data
        self.rise_data = pd.read_csv(running_folder + rise_data)
        self.country_mapping = pd.read_csv(running_folder + country_mapping)
        
        # Read in GDP per capita
        self.GDP_data = pd.read_csv(running_folder + GDP)
        self.GDP_data.columns = self.GDP_data.columns.map(str)
        self.GDP_change = pd.read_csv(running_folder + GDP_Change)
        self.GDP_change.columns = self.GDP_change.columns.map(str)
        
        
        # Add ISO 2 and 3 Codes (if required)
        self.generation_data = pd.merge(self.generation_data, self.country_codes, left_on='Country code', right_on='alpha-3')
        self.aures_waccs = pd.merge(self.aures_waccs, self.country_codes, left_on='Country', right_on='alpha-2')
        
        
    def pull_generation_data(self, year, technology, capacity=None):

        
        # Extract generation data
        generation_data = self.generation_data
        if capacity is not None:
            unit = "GW"
            category = "Capacity"
        else:
            unit = "%"
            category = "Electricity generation"

        
        # Extract specific year
        generation_subset = generation_data[(generation_data['Year'] == year) & (generation_data['Category'] == category) & (generation_data['Unit'] == unit)] 
                                                                                                             # Extract specific technology
        extracted_data = generation_subset[generation_subset['Variable'] == technology]
                                                                                                                                  
        # Extract specific columns that are required
        data_for_output = extracted_data[["Area", "Country code", "Year", "Continent", "Value","YoY absolute change"]]
        
        return data_for_output
    
    
    def pull_GDP_data(self, year):

        
        # Extract generation data
        data = self.GDP_data
        
        # Extract specific year
        data_subset = data[["Country code", year]]
           
        
        return data_subset
    
    
    def plot_scatter(self, x_data, y_data, z_data, c_data, text_data, x_label, y_label, ylog=None, xlog=None):
        
        plt.figure(figsize=(16, 12))
        plt.scatter(x_data, y_data, s=z_data, c=c_data)

        for i, txt in enumerate(text_data):
            plt.annotate(txt, (x_data[i], y_data[i]))
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if ylog is not None:
            plt.yscale('log')
        if xlog is not None:
            plt.xscale('log')

        plt.show()

        
    def plot_linear_relationship(self, data, parameter_1, parameter_2, remove_zeros=None, remove_country=None):

        # Filter Data
        filtered_data = data.dropna(subset=[parameter_1, parameter_2]).copy()
        filtered_data = data.dropna(subset=[parameter_1, parameter_2]).copy()
    
        if remove_zeros is not None:
            filtered_data = filtered_data[filtered_data[parameter_1] != 0]
            filtered_data = filtered_data[filtered_data[parameter_2] != 0]
    
        if remove_country is not None:
            filtered_data = filtered_data[filtered_data["Country code"] != remove_country]

        # Plotting
        plt.figure(figsize=(16, 12))

        # Scatter plot
        plt.scatter(filtered_data[parameter_1], filtered_data[parameter_2], color='blue', label='Predicted')
        X = filtered_data[[parameter_1]].values
        y = filtered_data[parameter_2].values
        model = LinearRegression().fit(X, y)
        m = model.coef_[0]
        b = model.intercept_
        plt.plot(X, m*X + b, color='red', label=f'Fitted line: y = {m:.2f}x + {b:.2f}')

        # Calculate R^2 value
        r2 = r2_score(y, model.predict(X))
        plt.text(0.95, 0.95, f'R^2 = {r2:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        plt.xlabel(parameter_1)
        plt.ylabel(parameter_2)

        # Adding country labels to data points
        for i, country in enumerate(filtered_data['Country code']):
            plt.text(filtered_data[parameter_1].iloc[i], filtered_data[parameter_2].iloc[i], country, color='black', fontsize=12, ha='center', va='top')

        plt.legend()
        plt.show()
        
        
    def produce_polynomial_regression_plots(self, data, selected_parameter, influencing_parameters, country_codes=None, print_values=None, variable_name=None):
    
        
        filtered_data = data.dropna(subset=influencing_parameters)
        filtered_data = filtered_data.dropna(subset=[selected_parameter])
    
        # Generate polynomial features
        poly = PolynomialFeatures(2)
        X_poly = poly.fit_transform(filtered_data[influencing_parameters])
    
        # Fit polynomial regression model
        poly_regr = linear_model.LinearRegression()
        poly_regr.fit(X_poly, filtered_data[selected_parameter])

    
        # Calculate R-squared value
        intercept = poly_regr.intercept_
        r_squared_poly = poly_regr.score(X_poly, filtered_data[selected_parameter])
        
        if print_values is not None:

            print("Intercept:", intercept)
            print("R-squared (Polynomial):", r_squared_poly)
            coefficients = poly_regr.coef_
            print("\nCoefficients:", coefficients)


        # Generate predicted values
        predicted_values_poly = poly_regr.predict(X_poly)

        # Plotting
        plt.figure(figsize=(16, 12))
        plt.scatter(filtered_data[selected_parameter], predicted_values_poly, color='blue', label='Predicted')
        plt.plot(filtered_data[selected_parameter].values, filtered_data[selected_parameter].values, color='red', linestyle='--', label='Actual')  # Plotting the line where predicted = actual
        if variable_name is None:
            variable_name = ""
        plt.xlabel('Actual ' + variable_name)
        plt.ylabel('Predicted ' + variable_name)
        plt.title('Actual vs Predicted ' + variable_name +' (Polynomial Regression)')
        plt.text(0.95, 0.95, f"R2 = {r_squared_poly:0.2f}", fontsize=12, ha='center', va='top') 
        
        if country_codes is not None:
            for i, country in enumerate(filtered_data['Country code']):
                plt.text(filtered_data[selected_parameter].iloc[i], predicted_values_poly[i], country, color='black', fontsize=12, ha='center', va='top')
        plt.legend()
        plt.show()
        
        
        
        
    def produce_linear_regression_plots_v3(self, data, selected_parameter, influencing_parameters, country_codes=None, print_values=None, variable_name=None, continent_color=None, return_values=None, filename=None, plot=None):
    
        filtered_data = data.dropna(subset=influencing_parameters)
        filtered_data = filtered_data.dropna(subset=[selected_parameter])
    
        # Set up linear regression
        regr = linear_model.Ridge(alpha=100)
        regr.fit(filtered_data[influencing_parameters], filtered_data[selected_parameter])
        
        # Assess coefficients 
        coefficients = regr.coef_
        coefficients_list = list(coefficients)
        intercept = regr.intercept_
        r_squared = regr.score(filtered_data[influencing_parameters], filtered_data[selected_parameter])
        
        # Print values, if desired
        if print_values is not None:
            print("Intercept:", intercept)
            print("Coefficients:", coefficients)
            print("R-squared:", r_squared)
        
    
        # Create estimates of predicted variables
        predicted_values = regr.predict(filtered_data[influencing_parameters])
    
        # Plotting 
        if plot is not None:
            fig, ax = plt.subplots(figsize=(16, 12))
            ax.scatter(filtered_data[selected_parameter], predicted_values, color='blue', label='Predicted')
            ax.plot(filtered_data[selected_parameter].values, filtered_data[selected_parameter].values, color='red', linestyle='--', label='Measure of Accuracy')
            if variable_name is None:
                variable_name = ""
            plt.xlabel('Actual ' + variable_name, fontsize=20)
            plt.ylabel('Predicted ' + variable_name, fontsize=20)
            plt.title('Actual vs Predicted ' + variable_name, fontsize=20)
            plt.text(0.5, 0.98, f"R2 = {r_squared:0.2f}", fontsize=20, ha='center', va='top', transform=ax.transAxes) 
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)


            color_dict = {'Africa':'black', 'South America':'green', 'Oceania':'brown', 'Europe':'blue', 'Asia':'cyan', 'North America':'red', np.nan:'yellow'}
            # Print country codes, if required
            if country_codes is not None:
                for i, country in enumerate(filtered_data['Country code']):
                    if continent_color is not None:    
                        color = color_dict[filtered_data.at[filtered_data.index[i],'Continent']]
                    else:
                        color = 'black'
                    ax.text(filtered_data[selected_parameter].iloc[i], predicted_values[i], country, color=color, fontsize=12, ha='center', va='top')
            plt.legend(fontsize=20)
            if filename is not None:
                plt.savefig(filename + "png")
            plt.show()
        
        if return_values is not None:
            return influencing_parameters, coefficients_list, intercept
        

    
    def extract_collated_data_v2(self, year, technology):
        
        # Extract RE data
        re_data = self.pull_generation_data_v2(year, technology)
        
        # Extract data on Interest Rates
        interest_rates = self.pull_IR_data(year)
        
        # Extract CRPs
        crp_data = self.pull_CRP_data(year)
        
        # Extract ERP
        erp_data = self.pull_CRP_data(year)
        
        # Extract GDP data
        gdp_data = self.pull_GDP_data(year)
        
        # Extract GDP data
        gdp_change = self.pull_GDP_change(year)
        
        # Extract Tax Rates
        tax_data = self.tax_data[['Country code', 'Tax_Rate']]
        
        # Merge all these data
        merged_df = pd.merge(crp_data, re_data, left_on="Country code", right_on="Country code", how="outer")
        merged_df = pd.merge(merged_df, gdp_data, left_on="Country code", right_on="Country code", how="outer")
        merged_df = pd.merge(merged_df, gdp_change, left_on="Country code", right_on="Country code", how="outer")
        merged_df = pd.merge(merged_df, interest_rates, left_on="Country code", right_on="Country code", how="outer")
        merged_df = pd.merge(merged_df, tax_data, left_on="Country code", right_on="Country code", how="outer")
        
        # Drop any rows without a CRP
        merged_df = merged_df.dropna(subset=["CRP_"+year])
        
        return merged_df
    
    def pull_IR_data(self, year):
                             
        # Extract IR data for a given year
        data_subset = self.collated_IR[["Country code", year]]
        
        # Rename the columns of the data
        data_subset = data_subset.rename(columns={year: "IR_"+year})
                             
        return data_subset
                             
    def pull_CRP_data(self, year):

        
        # Extract generation data
        data = self.crp_data
        
        # Extract specific year
        data_subset = data[["Country", "Country code", year]]
        data_subset = data_subset.rename(columns={year: "CRP_"+year})
        
        
        return data_subset
    
    
    def pull_GDP_data(self, year):

        
        # Extract generation data
        data = self.GDP_data
        
        # Extract specific year
        data_subset = data[["Country code", year]]
        data_subset = data_subset.rename(columns={year: "GDP_"+year})
        
        
        return data_subset
    
    def pull_GDP_change(self, year):

        
        # Extract generation data
        data = self.GDP_change
        
        # Extract specific year
        data_subset = data[["Country code", year]]
        data_subset = data_subset.rename(columns={year: "GDP_Change_"+year})
        
        
        return data_subset
        
    def pull_generation_data_v2(self, year_str, technology):

        
        # Extract generation data
        generation_data = self.generation_data
        year = int(year_str)
        
        # Extract Capacity
        capacity_subset = generation_data[(generation_data['Year'] == year) & (generation_data['Category'] == "Capacity") & (generation_data['Unit'] == "GW")]                                             
        capacity_data = capacity_subset[capacity_subset['Variable'] == technology]
        capacity_data = capacity_data.rename(columns = {"Value" : "Capacity_" + year_str, "YoY absolute change": "Capacity_" + year_str + "_YoY_Change"})

        
        # Extract Penetration
        penetration_subset = generation_data[(generation_data['Year'] == year) & (generation_data['Category'] ==  "Electricity generation") & (generation_data['Unit'] == "%")]  
        penetration_data = penetration_subset[penetration_subset['Variable'] == technology]
        penetration_data = penetration_data.rename(columns = {"Value" : "Penetration_" + year_str, "YoY absolute change": "Penetration_" + year_str + "_YoY_Change"})

        
        # Extract needed data
        penetration_data = penetration_data[["Area", "Country code", "Year", "Continent", "Penetration_" + year_str,"Penetration_" + year_str + "_YoY_Change"]]
        capacity_data  = capacity_data[["Country code", "Capacity_" + year_str, "Capacity_" + year_str + "_YoY_Change"]]
        data_for_output = pd.merge(penetration_data, capacity_data, on="Country code", how="outer")
        
        return data_for_output
    
    def estimate_irena_premiums(self, drop_data_gaps=None):
    
        # Extract all data for 2021
        collated_solar = self.extract_collated_data_v2("2021", "Solar")
        collated_wind = self.extract_collated_data_v2("2021", "Wind")    
        
        # Merge solar and wind data and rename column names
        collated_data = pd.merge(collated_solar[["Country code", "Capacity_2021", "Penetration_2021"]], collated_wind[["Country code", "Capacity_2021", "Penetration_2021"]], on="Country code", how="inner", suffixes=('_Solar', '_Wind'))
        collated_data = collated_data.rename(columns={"Penetration_2021_Solar":"Solar_Penetration_2021", "Penetration_2021_Wind":"Wind_Penetration_2021", "Capacity_2021_Solar":"Solar_Capacity_2021", "Capacity_2021_Wind":"Wind_Capacity_2021"}) 
    
        # Extract meta data and merge with the chosen
        merged_data = pd.merge(collated_solar[["Area", "Country code", "Continent", "GDP_2021", "GDP_Change_2021", "IR_2021", "CRP_2021", "Tax_Rate"]], collated_data[["Country code", "Wind_Capacity_2021", "Wind_Penetration_2021","Solar_Capacity_2021", "Solar_Penetration_2021"]], on="Country code", how="inner")

        
        # Pull the IRENA WACC data
        irena_waccs = self.irena_waccs.copy()
        irena_waccs[["Onshore Wind", "Offshore Wind", "Solar PV"]] = irena_waccs[["Onshore Wind", "Offshore Wind", "Solar PV"]]*100
        
        # Merge the relevant datasets
        irena_data = pd.merge(merged_data, irena_waccs, on="Country code", how="inner")
        
        # Merge in the cost of electricity
        irena_data = pd.merge(irena_data, self.electricity_prices, on="Country code", how="inner")
        
        # Calculate the debt share based on CRPs for each entry
        irena_data = self.get_debt_share(irena_data, "2021")
        
        # Extract the ERP for 2021 and the assumed debt lender margin
        crp = self.pull_CRP_data("2021")
        erp = crp[crp['Country code'] == 'ERP']
        erp_value = crp['CRP_2021'][0]
        debt_margin = 2
        self.debt_margin = debt_margin
        self.erp_value = erp_value
        
        # Calculate the common factor between equity and debt (risk free + country risk + technology risk + other risks)
        common_terms_wind = self.get_common_terms(irena_data, "Onshore Wind", erp_value, debt_margin)
        common_terms_solar = self.get_common_terms(irena_data, "Solar PV", erp_value, debt_margin)
        
        # Include common terms into the dataframe
        irena_data['Wind_Common_Risk_2021'] = common_terms_wind
        irena_data['Solar_Common_Risk_2021'] = common_terms_solar
        
        
        # Calculate cost of equity and cost of debt
        irena_data['Wind_Cost_Debt_2021'] = common_terms_wind + debt_margin
        irena_data['Wind_Cost_Equity_2021'] = common_terms_wind + erp_value - debt_margin
        irena_data['Solar_Cost_Debt_2021'] = common_terms_solar + debt_margin
        irena_data['Solar_Cost_Equity_2021'] = common_terms_solar + erp_value - debt_margin
        
        # Calculate the risk free rate 
        risk_free_rate_wind = np.nanmin(irena_data['Wind_Cost_Debt_2021'])
        risk_free_rate_solar = np.nanmin(irena_data['Solar_Cost_Debt_2021'])
        
        # Calculate the wind and solar technology premiums above the country risk and minimum risk rate (calculated as the minimum of the cost of debt)
        irena_data['Wind_Tech_Premium_2021'] = common_terms_wind - 0.5*irena_data['CRP_2021'] - risk_free_rate_wind
        irena_data['Solar_Tech_Premium_2021'] = common_terms_solar - 0.5*irena_data['CRP_2021'] - risk_free_rate_solar
        
        # Drop any lines without data on 
        if drop_data_gaps is not None:
            irena_data = irena_data.dropna(subset=["Onshore Wind", "Solar PV"], how='all') 
        irena_data = irena_data.rename(columns={"Onshore Wind":"Onshore_Wind_WACC_2021", "Solar PV": "Solar_WACC_2021", "Offshore Wind": "Offshore_Wind_WACC_2021"})
        
        # Add in RISE data
        rise_data = self.extract_rise_data(2021)
        irena_data = pd.merge(irena_data, rise_data, on="Country code", how="left")
        
        
        # Fill in missing RISE data
        irena_data[["RISE_1", "RISE_2", "RISE_3", "RISE_4", "RISE_5", "RISE_6", "RISE_7"]] = irena_data[["RISE_1", "RISE_2", "RISE_3", "RISE_4", "RISE_5", "RISE_6", "RISE_7"]].fillna(value=0)
        return irena_data
    
    
    def get_debt_share(self, data, year):
            
        # Define relationship that is being used between CRP and DEBT share
        data['Debt_Share_'+year] = (80 - 60) / (0 - np.nanmax(data['CRP_'+year])) * data['CRP_'+year] + 80
        
        return data
        
        
    def get_common_terms(self, data, technology, erp_value, debt_margin):
            
        # Set up values
        debt_share = data['Debt_Share_2021']/100
        tax_rate = 0
        wacc = data[technology]
        erp = erp_value
        drp = debt_margin
            
        # Apply the back calculation
        common_terms = (wacc - (1-debt_share)*erp - debt_share*(1-tax_rate)*drp ) / (1 - (tax_rate)*debt_share)
        # For any values returned negative, set to the lowest positive value
        common_terms_positive = common_terms
        min_positive_value = np.nanmin(np.where(common_terms_positive > 0, common_terms_positive, np.inf))
        common_terms[common_terms < 0] = min_positive_value    
        
        return common_terms
    
    
    
    
    
    
    def calculate_future_waccs_v2(self, influencing_parameters_wind, influencing_parameters_solar, year, interest_rate, percentage_increase=None):

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

        # Get irena 2021 data from calling separate function
        irena_data_extracted = self.estimate_irena_premiums()

        # Produce linear relationships using the selected parameters that were input
        parameters_wind, coefficients_wind, intercept_wind = self.produce_linear_regression_plots_v3(irena_data_extracted, "Wind_Common_Risk_2021", influencing_parameters_wind, country_codes=True, return_values=True, variable_name="Wind Common Risks (%)", print_values=True, filename="LR_Wind_Common_Costs")
        parameters_solar, coefficients_solar, intercept_solar = self.produce_linear_regression_plots_v3(irena_data_extracted, "Solar_Common_Risk_2021", influencing_parameters_solar, country_codes=True, return_values=True, variable_name="Solar Common Risks (%)", print_values=True, filename="LR_Solar_Common_Costs")


        # Estimate the deltas for each country between the linear regression and the values of debt for solar and wind. Start off with initial values
        delta_solar = irena_data_extracted['Solar_Common_Risk_2021']
        delta_wind = irena_data_extracted['Solar_Common_Risk_2021']
        predicted_solar = intercept_solar
        predicted_wind = intercept_wind

        # Using a loop, estimate the predicted and delta values
        for i, coefficient in enumerate(coefficients_solar):
            predicted_solar = predicted_solar + irena_data_extracted[parameters_solar[i]]*coefficients_solar[i]
            predicted_wind = predicted_wind + irena_data_extracted[parameters_wind[i]]*coefficients_wind[i]
            delta_solar = delta_solar - irena_data_extracted[parameters_solar[i]]*coefficients_solar[i]
            delta_wind = delta_wind - irena_data_extracted[parameters_wind[i]]*coefficients_wind[i]
        irena_data_extracted['Predicted_Solar_2021'] = predicted_solar
        irena_data_extracted['Predicted_Wind_2021'] = predicted_wind
        irena_data_extracted['Delta_Solar'] = delta_solar - intercept_solar  
        irena_data_extracted['Delta_Wind'] = delta_wind - intercept_wind
        
        # Calculate the error term
        irena_data_extracted['Error_Solar'] = irena_data_extracted['Solar_Common_Risk_2021'] - irena_data_extracted['Predicted_Solar_2021']
        irena_data_extracted['Error_Wind'] = irena_data_extracted['Wind_Common_Risk_2021'] - irena_data_extracted['Predicted_Wind_2021']
        
        # Estimate the change in risk free rate globally
        US_IR = self.OECD_IR[self.OECD_IR['Country code'] == "USA"]
        risk_free_rate_increase = interest_rate - US_IR['2021'].values[0]
        irena_data_extracted['IR_Change_'+year + '_2021'] = risk_free_rate_increase

        # Estimate the base rate for each country
        irena_data_extracted['Intercept_Solar_'+year] = intercept_solar 
        irena_data_extracted['Intercept_Wind_'+year] = intercept_wind 
        
        # Calculate number of years
        num_years = int(year)-2022
        
        # Get generation data for the specified year
        re_data_solar = self.pull_generation_data_v2(year, "Solar")
        re_data_wind = self.pull_generation_data_v2(year, "Wind")
        
        # Fill missing datapoints with 2022 / 2023 data. 
        re_solar_2022 = self.pull_generation_data_v2("2022", "Solar")
        re_wind_2022 = self.pull_generation_data_v2("2022", "Wind")
        re_data_solar = fill_missing_RE_values(re_data_solar, re_solar_2022, "2022")
        re_data_wind = fill_missing_RE_values(re_data_wind, re_wind_2022, "2022")
        re_solar_2021 = self.pull_generation_data_v2("2021", "Solar")
        re_wind_2021= self.pull_generation_data_v2("2021", "Wind")
        re_data_solar = fill_missing_RE_values(re_data_solar, re_solar_2021, "2021")
        re_data_wind = fill_missing_RE_values(re_data_wind, re_wind_2021, "2021")
        
        # Calculate the new generation
        if int(year) > 2023:
            re_data_solar[["Capacity_"+year,"Penetration_"+year]] = re_data_solar[["Capacity_2022","Penetration_2022"]] * (1+ percentage_increase) ^ num_years
            re_data_wind[["Capacity_"+year,"Penetration_"+year]] = re_data_wind[["Capacity_2022","Penetration_2022"]] * (1+ percentage_increase) ^ num_years


        # Collate the wind and solar generation data for the specified year
        collated_renewables_data = pd.merge(re_data_solar[["Country code", "Capacity_"+year, "Penetration_"+year]], re_data_wind[["Country code", "Capacity_"+year, "Penetration_"+year]], on="Country code", how="outer", suffixes=('_Solar', '_Wind'))
        collated_renewables_data = collated_renewables_data.rename(columns={"Penetration_"+year + "_Solar":"Solar_Penetration_"+year, "Penetration_"+year + "_Wind":"Wind_Penetration_"+year, "Capacity_"+year + "_Solar":"Solar_Capacity_"+year, "Capacity_"+year + "_Wind":"Wind_Capacity_"+year}) 

        # Set up a storage dataframe for countries with renewable energy generation statistics for the specific year
        storage_dataframe = collated_renewables_data.copy()


        # Extract CRPs
        crp_data = self.pull_CRP_data("2023")
        crp_data.rename(columns={"CRP_2023":"CRP_" + year})

        # Merge the CRP and IR data into the storage dataframe
        storage_dataframe = pd.merge(crp_data,  storage_dataframe, on="Country code", how="left")

        # Merge the storage dataframe with a copy of the irena_extracted_data
        irena_data_extracted = irena_data_extracted.drop(labels={"Country"},axis=1, errors="ignore")
        collated_data_to_merge = irena_data_extracted.copy()
        storage_dataframe = pd.merge(storage_dataframe, collated_data_to_merge.copy(), on="Country code", how="inner")
        storage_dataframe.set_index('Country code')
        collated_data_to_merge.set_index('Country code')

        # Rename the wind and solar parameters to reflect the new year
        parameters_wind= [item.replace('_2021', '_'+year) for item in parameters_wind]
        parameters_solar= [item.replace('_2021', '_'+year) for item in parameters_solar]
        
        # Calculate the wind cost of debt for the new year, using a loop over the input parameters
        storage_dataframe['Wind_Cost_Debt_' + year]= storage_dataframe['Intercept_Wind_'+year]
        for i, parameter in enumerate(parameters_wind):
            storage_dataframe['Wind_Cost_Debt_' + year] = storage_dataframe['Wind_Cost_Debt_' + year] + coefficients_wind[i]*storage_dataframe[parameter]
        storage_dataframe['Wind_Common_Risk_' + year] = storage_dataframe['Wind_Cost_Debt_' + year] - self.debt_margin
        
        # Set negative predictions for CR to the minimum positive value
        positive_CR_wind = np.nanmin(np.where(storage_dataframe['Wind_Common_Risk_'+year].copy().values > 0, storage_dataframe['Wind_Common_Risk_'+year].copy().values, np.inf))
        storage_dataframe.loc[storage_dataframe['Wind_Common_Risk_'+year] < 0, 'Wind_Common_Risk_'+year] = positive_CR_wind
        
        # Add in debt margin and change in interest rates to get cost of debt for wind
        storage_dataframe['Wind_Cost_Debt_' + year] = storage_dataframe['Wind_Common_Risk_' + year] + irena_data_extracted['IR_Change_'+year + '_2021'] + self.debt_margin

        # Calculate the solar cost of debt for the new year, using a loop over the input parameters    
        storage_dataframe['Solar_Cost_Debt_' + year]= storage_dataframe['Intercept_Solar_'+year]
        for i, parameter in enumerate(parameters_wind):
            storage_dataframe['Solar_Cost_Debt_' + year] = storage_dataframe['Solar_Cost_Debt_' + year] + coefficients_solar[i]*storage_dataframe[parameter]
        storage_dataframe['Solar_Common_Risk_' + year] = storage_dataframe['Solar_Cost_Debt_' + year] - self.debt_margin
        
        # Set negative predictions for CR to the minimum positive value
        positive_CR_solar = np.nanmin(np.where(storage_dataframe['Solar_Common_Risk_' + year].copy().values > 0, storage_dataframe['Solar_Common_Risk_' + year].copy().values, np.inf))
        storage_dataframe.loc[storage_dataframe['Solar_Common_Risk_' + year] < 0, 'Solar_Common_Risk_' + year] = positive_CR_solar
        
        # Add in debt margin and change in interest rates to get cost of debt for solar
        storage_dataframe['Solar_Cost_Debt_' + year] = storage_dataframe['Solar_Common_Risk_' + year] + irena_data_extracted['IR_Change_'+year + '_2021'] + self.debt_margin


        # Calculate equity cost using ERP
        erp = crp_data[crp_data['Country code'] == 'ERP']['CRP_'+year].values[0]
        print(erp)
        storage_dataframe['Wind_Cost_Equity_' + year] = storage_dataframe['Wind_Cost_Debt_' + year] + erp - self.debt_margin
        storage_dataframe['Solar_Cost_Equity_' + year] = storage_dataframe['Solar_Cost_Debt_' + year] + erp - self.debt_margin

        # Get new debt share
        storage_dataframe = self.get_debt_share(storage_dataframe, year)

        # Calculate the WACC for Wind and Solar based on Tax Rates and Debt Shares
        storage_dataframe["Onshore_Wind_WACC_"+year] = (storage_dataframe['Wind_Cost_Debt_' + year] * (1 - storage_dataframe['Tax_Rate']/100) * storage_dataframe['Debt_Share_'+year]/100) + storage_dataframe['Wind_Cost_Equity_' + year]*(1-storage_dataframe['Debt_Share_'+year]/100)
        storage_dataframe["Solar_WACC_"+year] = (storage_dataframe['Solar_Cost_Debt_' + year] * (1 - storage_dataframe['Tax_Rate']/100) * storage_dataframe['Debt_Share_'+year]/100) +  (storage_dataframe['Solar_Cost_Equity_' + year]*(1-storage_dataframe['Debt_Share_'+year]/100))

        # Sort storage_dataframe
        storage_dataframe = pd.merge(storage_dataframe, self.country_mapping, on="Country code", how="inner")
        first_cols = ['Country', 'Country code', 'Continent']
        other_cols = np.sort(storage_dataframe.columns.difference(first_cols)).tolist()
        storage_dataframe = storage_dataframe.loc[:, first_cols+other_cols]
        storage_dataframe = storage_dataframe.drop(columns={"Country", "Area", "Electricity_Price", "GDP_2021", "GDP_Change_2021"})
        storage_dataframe = storage_dataframe.dropna(subset={"CRP_"+year})
        extracted_data = storage_dataframe[["Country Name", "Country code", "index", "Tax_Rate", "CRP_" + year, "CRP_2021", 'IR_Change_'+year + '_2021', "Debt_Share_" + year, "Debt_Share_2021", "Onshore_Wind_WACC_" + year,  "Onshore_Wind_WACC_2021","Solar_WACC_" + year, "Solar_WACC_2021",  "Wind_Cost_Debt_" + year, "Wind_Cost_Equity_" + year, "Wind_Common_Risk_" + year, "Wind_Cost_Debt_2021", "Wind_Cost_Equity_2021", "Wind_Common_Risk_2021", "Wind_Penetration_" + year, "Wind_Penetration_2021", "Delta_Wind", "Error_Wind", "Solar_Cost_Debt_" + year, "Solar_Cost_Equity_" + year, "Solar_Common_Risk_" + year, "Solar_Cost_Debt_2021", "Solar_Cost_Equity_2021","Solar_Common_Risk_2021", "Solar_Penetration_" + year, "Solar_Penetration_2021", "Intercept_Solar_"+year, "Delta_Solar", "Error_Solar"]]
        
        # Produce dataframe with estimated waccs for LCOE modelling
        estimated_country_waccs = pd.merge(self.country_mapping, extracted_data[['index', 'Onshore_Wind_WACC_' + year, "Solar_WACC_" + year]], how="left", on="index")
        estimated_country_waccs = estimated_country_waccs.rename(columns={"Onshore_Wind_WACC_" + year: "onshore_wacc", "Solar_WACC_" + year: "solar_pv_wacc"})
        

        return extracted_data, estimated_country_waccs
    
    
    def extract_rise_data(self, year):
        
        rise_data = self.rise_data
        
        if year > 2021:
            # Get parameters
            parameters = ['RISE_1', 'RISE_2', 'RISE_3', 'RISE_4', 'RISE_5', 'RISE_6', 'RISE_7']
            non_parameter_columns = ['Country', 'Country code']  

            # Get data for 2021 and 2016 for interpolating
            data_2021 = rise_data[rise_data['Year'] == 2021].reset_index()
            data_2019 = rise_data[rise_data['Year'] == 2019].reset_index()

            # Interpolate data
            projected_data = data_2021[parameters] + (data_2021[parameters] - data_2019[parameters]) * (int(year) - int(2019))
            projected_data[projected_data < data_2021[parameters]] = data_2021[parameters]
            projected_data = projected_data.clip(upper=100)
            projected_data['RISE_OVERALL'] = projected_data.sum(axis=1) * 1 / 7 # Dividing by seven as each of the seven RISE scores are out of 100

            # Add in non-parameter columns
            non_parameter_data = data_2021[non_parameter_columns]

            # Concatenate the non-parameter columns with the selected_data
            selected_data = pd.concat([non_parameter_data.reset_index(drop=True), projected_data.reset_index(drop=True)], axis=1)

        else:
            # Select the data 
            selected_data = rise_data[rise_data['Year'] == year]
            selected_data = selected_data.drop(columns={"Year"})
            
        # Set RISE scores to zero for all additional countries
        selected_data = pd.merge(self.country_mapping[['Country code']], selected_data, on="Country code", how="left")
        selected_data = selected_data.fillna(value=0)

        return selected_data
    
    


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


    def estimate_wacc(self, data, year, technology, erp=None):


        # Get Debt and Equity Risk Premiums for the year
        if int(year) > 2023:
            equity_risk = erp
        else:
            crp_data = self.pull_CRP_data(str(year))
            erp = crp_data[crp_data["Country code"] == "ERP"]["CRP_"+year].values[0]

        # Calculate Debt Share
        data = self.get_debt_share(data, year)
        data['Debt_Share_'+year] = data['Debt_Share_'+year]/100

        # Calculate Cost of Debt
        data[technology + "_Cost_Debt_" + year] = data[technology + "_Common_Risk_" + year] + self.debt_margin
        data[technology + "_Cost_Debt_LB_" + year] = data[technology + "_Common_Risk_LB_" + year] + self.debt_margin
        data[technology + "_Cost_Debt_UB_" + year] = data[technology + "_Common_Risk_UB_" + year] + self.debt_margin

        # Calculate Cost of Debt
        data[technology + "_Cost_Equity_" + year] = data[technology + "_Common_Risk_" + year] + erp
        data[technology + "_Cost_Equity_LB_" + year] = data[technology + "_Common_Risk_LB_" + year] + erp
        data[technology + "_Cost_Equity_UB_" + year] = data[technology + "_Common_Risk_UB_" + year] + erp

        # Calculate WACC
        data[technology + "_WACC_" + year] = data[technology + "_Cost_Debt_" + year] * data["Debt_Share_" + year] * (1 - data["Tax_Rate"]/100) + (1 - data["Debt_Share_" + year]) * data[technology + "_Cost_Equity_" + year]
        data[technology + "_WACC_LB_" + year] = data[technology + "_Cost_Debt_LB_" + year] * data["Debt_Share_" + year] * (1 - data["Tax_Rate"]/100) + (1 - data["Debt_Share_" + year]) * data[technology + "_Cost_Equity_LB_" + year]
        data[technology + "_WACC_UB_" + year] = data[technology + "_Cost_Debt_UB_" + year] * data["Debt_Share_" + year] * (1 - data["Tax_Rate"]/100) + (1 - data["Debt_Share_" + year]) * data[technology + "_Cost_Equity_UB_" + year]

        return data

    def collate_future_data(self, year):

        # Get CRPs
        if int(year) > 2023:
            crp_data = self.pull_CRP_data("2023")
            crp_data.rename(columns={"CRP_2023":"CRP_" + year})
        else: 
            crp_data = self.pull_CRP_data(str(year))

        # Get Renewable Penetration
        def fill_missing_RE_values(data, previous_year, year):

            # Set Country Code as index
            data.set_index('Country code', inplace=True)
            previous_year.set_index('Country code', inplace=True)

            # Fill missing values for 2023 with 2022 data
            data = pd.merge(data, previous_year, on="Country code", how="outer")
            data['Penetration_'+year] = data['Penetration_'+year].fillna(data['Penetration_'+str(int(year)-1)])

            # Reset index if needed
            data.reset_index(inplace=True)

            return data

        def drop_y(df):
            # list comprehension of the cols that end with '_y'
            to_drop = [x for x in df if x.endswith('_y')]
            df.drop(to_drop, axis=1, inplace=True)

         # Get generation data for the specified year
        re_data_solar = self.pull_generation_data_v2(year, "Solar")
        re_data_wind = self.pull_generation_data_v2(year, "Wind")

        # Fill missing datapoints with 2022 data. 
        if int(year) > 2022:
            re_solar_2022 = self.pull_generation_data_v2("2022", "Solar")
            re_wind_2022 = self.pull_generation_data_v2("2022", "Wind")
            re_data_solar = fill_missing_RE_values(re_data_solar, re_solar_2022, "2023")
            re_data_wind = fill_missing_RE_values(re_data_wind, re_wind_2022, "2023")

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
        rise_data = self.extract_rise_data(int(year))
        rise_data = rise_data.rename(columns={c:c+"_" + year for c in rise_data.columns if "RISE" in c})

        # Collate data
        collated_data = pd.merge(crp_data, collated_renewables_data, how="left", on="Country code")
        collated_data = pd.merge(collated_data, rise_data, how="left", on="Country code", suffixes=('', '_y'))
        drop_y(collated_data)
        future_data = collated_data                  


        return future_data

    def estimate_future_waccs_v4(self, selected_parameter_wind, selected_parameter_solar, influencing_parameters_wind, influencing_parameters_solar, year, interest_rate=None, percentage_increase=None):


        # Get 2021 data from calling separate function
        irena_data_extracted = self.estimate_irena_premiums()
        irena_data_extracted = irena_data_extracted.rename(columns={c:c+"_2021" for c in irena_data_extracted.columns if "RISE" in c})
        irena_data_extracted = irena_data_extracted.rename(columns={"Wind_Common_Risk_2021": "Onshore_Wind_Common_Risk_2021", "Wind_Cost_Equity_2021":"Onshore_Wind_Cost_Equity_2021", "Wind_Cost_Debt_2021":"Onshore_Wind_Cost_Debt_2021"})

        # Calculate 2021 risk-free rate
        US_IR = self.OECD_IR[self.OECD_IR['Country code'] == "USA"]
        rf_rate_2021 =  US_IR['2021'].values[0]

        # Get desired year of data
        future_data = self.collate_future_data(year)

        # Calculate desired year of data risk free rate
        if int(year) > 2023:
            rf_rate = interest_rate
        else:
            rf_rate =  US_IR[year].values[0]

        # Risk free rate change
        rf_change = rf_rate - rf_rate_2021

        # Call the regression function with the corresponding parameters
        solar_estimates = self.run_regression_model(irena_data_extracted, future_data, selected_parameter_solar, influencing_parameters_solar, ["Solar_Penetration_"+year, "CRP_" + year, "RISE_OVERALL_"+year], year, rf_change, "Solar", print_values="True", filename=None, plot=None)
        wind_estimates = self.run_regression_model(irena_data_extracted, future_data, selected_parameter_wind, influencing_parameters_wind, ["Wind_Penetration_"+year, "CRP_" + year, "RISE_OVERALL_"+year], year, rf_change, "Onshore_Wind", print_values="True", filename=None, plot=None)

        # Call function to estimate debt share, cost of debt, cost of equity and WACC
        solar_estimated_data = self.estimate_wacc(solar_estimates, year,  "Solar")
        wind_estimated_data = self.estimate_wacc(wind_estimates, year, "Onshore_Wind")

        # Aggregated estimated data
        extracted_data = pd.merge(solar_estimated_data, wind_estimated_data, how="left", on="Country code", suffixes=("", "_y"))
        extracted_data.drop([col for col in extracted_data.columns if col.endswith('_y')],axis=1,inplace=True)
        extracted_data = pd.merge(extracted_data, self.country_mapping,  how="left", on = "Country code")
        
        # Add in estimated for offshore wind
        offshore_wind_premium = 1.73333
        extracted_data['Offshore_Wind_WACC_' + year] = extracted_data['Onshore_Wind_WACC_' + year] + offshore_wind_premium

        # Extract waccs for onshore, offshore and solar PV
        estimated_country_waccs = pd.merge(self.country_mapping, extracted_data[['Country code', 'Onshore_Wind_WACC_' + year, 'Offshore_Wind_WACC_' + year, "Solar_WACC_" + year]], how="left", on="Country code")
        estimated_country_waccs = estimated_country_waccs.rename(columns={"Onshore_Wind_WACC_" + year: "onshore_wacc", "Solar_WACC_" + year: "solar_pv_wacc", "Offshore_Wind_WACC_" + year: "offshore_wacc"})
        

        return extracted_data, estimated_country_waccs


    
    
    


                                      
        
        
 
                                      