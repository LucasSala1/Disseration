import pandas as pd

from .country import Country

def get_WTO_countries(year):
    """
    construct a list of countries from the WTO data for a particular year
    
    parameters:
        - year: int
    
    side effects:
        - gets data from csv-files, `captial_dict` and `area_dict`
    
    returns:
        - list of countries
        
    """
    
    df_e = pd.read_csv('country_data_e.csv', sep=';', na_values=[""])
    df_i = pd.read_csv('country_data_i.csv', sep=';', na_values=[""])
    
    series_i = clean_and_get_year(df_i, year).rename('i')
    series_e = clean_and_get_year(df_e, year).rename('e')
    df = pd.concat([series_i, series_e], axis=1, sort=True)
    
    incomplete_data_df = df[pd.isnull(df).any(axis=1)]
    if len(incomplete_data_df) > 0:
        print('for the following countries your data is incomplete:')
        print(incomplete_data_df)
        
    df = df.dropna()
    
    countries = []
    for name in df.index:
        country = Country(
            name,
            df['e'].at[name], 
            df['i'].at[name], 
        )
        
        countries.append(country)
        
    return countries

def clean_and_get_year(df,year):
    clean_df = df.rename(columns={'Unnamed: 0': 'Country'}).set_index('Country')
    year = str(year)
    return clean_df[year].dropna()
