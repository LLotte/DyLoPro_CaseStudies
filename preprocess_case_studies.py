"""
Contains the preprocessing pipeline for the event logs examined in each 
case study. 
"""
from asyncio import log
import pandas as pd 
import numpy as np

def preprocess_bpic19(log):
    """Preprocess the bpic19 event log. 

    Parameters
    ----------
    log : pandas.DataFrame 
        Event log.

    Returns
    -------
    log : pandas.DataFrame
        Preprocessed event log.
    """
    # Convert timestamp column to datetime64[ns, UTC] dtype 
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], format='%Y-%m-%d %H:%M:%S%z').dt.tz_convert('UTC')

    # Convert 2 cols to object dtype. 
    log['case:Item'] = log['case:Item'].astype('object')
    log['case:Purchasing Document'] = log['case:Purchasing Document'].astype('object')
    
    return log 

def preprocess_bpic12(log):
    """Preprocess the BPIC12 event log for DyLoPro."""

    import numpy as np
    import pandas as pd

    log = log.copy()

    # --- Timestamps ---
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], errors='coerce', utc=True)
    log = log.dropna(subset=['time:timestamp', 'concept:name', 'case:concept:name'])

    # --- Categorical columns ---
    for col in ['case:concept:name', 'org:resource', 'concept:name', 'lifecycle:transition']:
        if col in log.columns:
            log[col] = log[col].astype('object')

    # --- Trace-level features ---
    if 'case:AMOUNT_REQ' in log.columns:
        log['case:AMOUNT_REQ'] = pd.to_numeric(log['case:AMOUNT_REQ'], errors='coerce')
    if 'case:REG_DATE' in log.columns:
        log['case:REG_DATE'] = pd.to_datetime(log['case:REG_DATE'], errors='coerce', utc=True)

    # --- Case duration ---
    case_durations = (
        log.groupby('case:concept:name')['time:timestamp']
        .agg(['min', 'max'])
        .assign(case_duration=lambda x: (x['max'] - x['min']).dt.total_seconds())
    )
    log = log.merge(case_durations['case_duration'], on='case:concept:name', how='left')

    # --- Time since case start & inter-event time ---
    log['time_since_case_start'] = (
        log['time:timestamp'] - log.groupby('case:concept:name')['time:timestamp'].transform('min')
    ).dt.total_seconds()
    log['inter_event_time'] = log.groupby('case:concept:name')['time:timestamp'].diff().dt.total_seconds()

    # --- DyLoPro outcomes ---
    # Step 1: Compute the last activity per case
    last_event_per_case = log.groupby('case:concept:name')['concept:name'].last()
    
    # Step 2: Map last activity to all events in the case
    log['last_event'] = log['case:concept:name'].map(last_event_per_case)

    # Step 3: Generate binary outcome columns only for events that exist in the data
    unique_last_events = last_event_per_case.unique()

    if 'A_APPROVED' in unique_last_events:
        log['case_approved'] = np.where(log['last_event'] == 'A_APPROVED', 1, 0).astype(int)

    if 'A_DECLINED' in unique_last_events:
        log['case_declined'] = np.where(log['last_event'] == 'A_DECLINED', 1, 0).astype(int)

    if 'A_CANCELLED' in unique_last_events:
        log['case_cancelled'] = np.where(log['last_event'] == 'A_CANCELLED', 1, 0).astype(int)

    # Step 4: Optional categorical outcome column
    log['case:outcome'] = log['last_event']

    # Step 5: Drop helper column
    log = log.drop(columns=['last_event'])

    return log


def preprocess_bpic17(log):
    """Preprocess the bpic17 event log. 

    Parameters
    ----------
    log : pandas.DataFrame 
        Event log.

    Returns
    -------
    log : pandas.DataFrame
        Preprocessed event log.
    """

    # Inner function to label (1-0) for 3 new target cols
    def _labeling(row, target):
        if row['last_o_act'] == target:
            return 1
        else:
            return 0

    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], format = 'mixed').dt.tz_convert('UTC')

    # Specifying which activities (if occurring) last indicate whether a case is ...
    relevant_offer_events = ["O_Accepted", "O_Refused", "O_Cancelled"]

    # Retaining only the Offer events 
    log_offer_events = log[log['EventOrigin'] == "Offer"]

    # Getting a dataframe that gives the last Offer activity for each case. 
    last_Offer_Activities = log_offer_events.groupby('case:concept:name', sort=False).last().reset_index()[['case:concept:name','concept:name']]
    last_Offer_Activities.columns = ['case:concept:name', 'last_o_act']

    # Adding that column as a case feature to the main log by merging on case:concept:name: 
    log = log.merge(last_Offer_Activities, on = 'case:concept:name', how = 'left')

    # Subsetting last_Offer_Activities dataframe for only the invalid cases. 
    last_Offer_Activities_invalid = last_Offer_Activities[~last_Offer_Activities['last_o_act'].isin(relevant_offer_events)]

    invalid_cases_list = list(last_Offer_Activities_invalid['case:concept:name'])

    # Dropping all invalid cases (and their events) from the main event log 
    log = log[~log['case:concept:name'].isin(invalid_cases_list)]

    # Adding the three 1-0 target columns 'case accepted', 'case refused', 'case canceled'

    # and adding another categorical case feature that contains 3 levels, indicating whether 
    # a case is 'Accepted', 'Refused' or 'Canceled':
    log['case:outcome'] = log['last_o_act'].copy()
    categorical_outcome_labels = ['Accepted', 'Refused', 'Canceled']
    binary_outcome_colnames = ['case accepted', 'case refused', 'case canceled']
    for idx in range(3):
        offer_event = relevant_offer_events[idx]
        out_label = categorical_outcome_labels[idx]
        out_colname = binary_outcome_colnames[idx]
        log['case:outcome'] = np.where(log['last_o_act'] == offer_event, out_label, log['case:outcome'])
        log[out_colname] = np.where(log['last_o_act'] == offer_event, 1, 0)
    
    log = log.drop(['last_o_act'], axis = 1)

    return log 

def preprocess_RTFM(log):
    """Preprocess the Road Traffic Fines Management event log. 

    Parameters
    ----------
    log : pandas.DataFrame 
        Event log.

    Returns
    -------
    log : pandas.DataFrame
        Preprocessed event log.
    """

    # Converting the appropriate columns to object dtype 
    to_object = {'org:resource': object, 'article': object, 'points': object}

    # Convert columns to the specified dtypes
    log = log.astype(to_object)

    # Covnert timestamp col to appropriate datetime format 
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], utc=True)

    # Drop the 'matricola' column
    log = log.drop('matricola', axis=1)

    return log 
