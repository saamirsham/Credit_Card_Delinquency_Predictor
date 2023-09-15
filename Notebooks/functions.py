import pandas as pd
import numpy as np


def n_mo_delinquency(data, months_list):
    """
    Calculate the delinquency status for a given number of months.

    Parameters:
        data (DataFrame): Input data containing 'ID', 'MONTHS_BALANCE', and 'is_delinquent' columns.
        months_list (list): List of integers specifying the number of months to consider.

    Returns:
        DataFrame: Updated data with additional columns indicating delinquency status for each specified number of months.
    """
    # Calculate the rank of each row within each group
    data['month_rank'] = data.groupby('ID')['MONTHS_BALANCE'].transform('rank')

    for num in months_list:
        # Get the top 'num' ranks for each ID
        nlargest = data.groupby('ID')['month_rank'].nlargest(num).reset_index(level=1, drop=True).reset_index()

        # Merge the selected rows with the original data to get the recent 'num' credit reports
        recent_n_credit_reports = data.merge(nlargest, how='inner', on=['ID', 'month_rank'])

        # Calculate the maximum delinquency status within the recent 'num' credit reports for each ID
        recent_n = recent_n_credit_reports.groupby('ID')['is_delinquent'].max().rename(f'{num}mo_delinquency')

        # Merge the delinquency information with the original data
        data = data.merge(recent_n, how='left', on='ID')

    return data

############################

def credit_approval_data_cleaner(credit_data, application_data, months_list):
    """
    Clean and merge credit approval data with application data.

    Parameters:
        credit_data (DataFrame): Credit data containing 'ID', 'STATUS', 'MONTHS_BALANCE', 'is_delinquent' columns.
        application_data (DataFrame): Application data containing 'ID', 'CODE_GENDER', 'OCCUPATION_TYPE' columns.
        months_list (list): List of integers specifying the number of months to consider for delinquency.

    Returns:
        DataFrame: Cleaned and merged data containing relevant credit and application information.
    """
    # Clean credit data
    credit_data = credit_data.copy()
    credit_data['is_delinquent'] = np.where(credit_data['STATUS'].isin(['C', 'X']), 0, 1)
    credit_data = credit_data.drop(columns='STATUS')
    credit_data['length_of_credit'] = credit_data.groupby('ID')['MONTHS_BALANCE'].transform('count')
    credit_data['number_of_delinquent_months'] = credit_data.groupby('ID')['is_delinquent'].transform('sum')
    credit_data['average_delinquency_rate'] = credit_data.groupby('ID')['is_delinquent'].transform('mean')

    # Apply delinquency calculation for specified number of months
    credit_full = n_mo_delinquency(credit_data.set_index('ID'), months_list)

    # Sort, remove duplicates, and drop unnecessary columns
    credit_full = credit_full.reset_index().sort_values(['ID', 'MONTHS_BALANCE'], ascending=[True, False])
    credit_cleaned = credit_full.drop_duplicates(subset='ID', keep='first').drop(columns=['MONTHS_BALANCE', 'month_rank'])

    # Clean application data
    application_data = application_data.drop(columns='CODE_GENDER')
    application_data['OCCUPATION_TYPE'].fillna('missing', inplace=True)
    
    application_data['DAYS_EMPLOYED'] = np.where(
        application_data['DAYS_EMPLOYED'] > 0, 0, application_data['DAYS_EMPLOYED'])
    
    application_data['OCCUPATION_TYPE'] = np.where(
        application_data['DAYS_EMPLOYED'] == 0, 'Retired', application_data['OCCUPATION_TYPE'])
    
    application_data['AGE'] = (abs(application_data['DAYS_BIRTH']) / 365).astype(int)
    
    application_data['YEARS_EMPLOYED'] = (abs(application_data['DAYS_EMPLOYED']) / 365).astype(int)
    
    application_data.drop(columns = ['DAYS_BIRTH', 'DAYS_EMPLOYED'], inplace = True)


    # Merge cleaned credit and application data
    cridit_new = credit_cleaned.merge(application_data, how='inner', on='ID')
    
    cridit_new.columns = cridit_new.columns.str.lower()
    
    return cridit_new
    