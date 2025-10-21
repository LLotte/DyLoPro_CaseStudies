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
    # Covnert timestamp col to appropriate datetime format 
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], utc=True)

    #ADDED
    # Drop unnecessary columns
    for col in ['matricola', 'lifecycle:transition']:
        if col in log.columns:
            log = log.drop(col, axis=1)

    # Sort events by case and timestamp
    log = log.sort_values(['case:concept:name', 'time:timestamp'])



    # --- CASE-LEVEL CATEGORICAL FEATURES ---
    categorical_case_cols = ['notificationType', 'lastSent', 'vehicleClass', 'article', 'points']

    for col in categorical_case_cols:
        if col not in log.columns:
            continue

        if col in ['notificationType', 'lastSent']:
            # Fill with the first non-null value per case, or "None" if missing entirely
            log[col] = (
                log.groupby('case:concept:name')[col]
                   .transform(lambda x: x.dropna().iloc[0] if x.notna().any() else 'None')
                   .astype(object)
            )
        else:
            # For columns expected to stay constant (e.g. vehicleClass, article, points)
            log[col] = (
                log.groupby('case:concept:name')[col]
                   .ffill()
                   .bfill()
                   .astype(object)
            )


    
    # --- ADD DISMISSAL OUTCOME VARIABLES ---
    # Get the last dismissal value per case (the final outcome)
    dismissal_case = (
        log.groupby('case:concept:name')['dismissal']
        .last()
    )
    # Create binary outcome
    log = log.merge(
        dismissal_case.rename('dismissal_case'),
        left_on='case:concept:name',
        right_index=True,
        how='left'
    )
    log['outcome_dismissed'] = ((log['dismissal_case'] == '#') | (log['dismissal_case'] == 'G')).astype(int)
    # Drop the dismissal_case column
    log = log.drop('dismissal_case', axis=1)

    # Numeric case feature
    # if 'expense' in log.columns:
    #     log['expense'] = log.groupby('case:concept:name')['expense'].transform(
    #         lambda x: x.ffill().bfill()
    #     )

    if 'expense' in log.columns:
        log['expense'] = (
            log.groupby('case:concept:name')['expense']
            .transform(lambda x: x.ffill().bfill())
            .fillna(0)
        )

    if 'org:resource' in log.columns:
        log['org:resource'] = (
            log.groupby('case:concept:name')['org:resource']
            .ffill()
        )

    # Clean and convert categorical columns to string without '.0' and replace missing values
    for col in ['org:resource', 'article', 'points']:
        log[col] = (
            log[col]
            .apply(lambda x: str(int(x)) if pd.notnull(x) and isinstance(x, (int, float)) and float(x).is_integer() 
                else (str(x) if pd.notnull(x) else 'Unknown')))
    


    # --- ADD PAYMENT OUTCOME VARIABLES ---
    # Last non-null amount per case
    last_amount = (
        log.dropna(subset=['amount'])
           .groupby('case:concept:name')['amount']
           .last()
           .rename('last_amount')
    )

    # Last non-null totalPaymentAmount per case
    last_total = (
        log.dropna(subset=['totalPaymentAmount'])
           .groupby('case:concept:name')['totalPaymentAmount']
           .last()
           .rename('last_totalPaymentAmount')
    )

    # Merge back to the main DataFrame
    log = log.merge(last_amount, on='case:concept:name', how='left')
    log = log.merge(last_total, on='case:concept:name', how='left')

    # Compute outcome_payment
    def payment_outcome(row):
        total = row['last_totalPaymentAmount']
        amount = row['last_amount']
        if pd.isna(total) or total == 0:
            return 'unpaid'
        elif total >= amount:
            return 'fully_paid'
        else:
            return 'partially_paid'

    log['outcome_payment'] = log.apply(payment_outcome, axis=1)

    log['outcome_fully_paid'] = (log['outcome_payment'] == 'fully_paid').astype(int)
    log['outcome_partially_paid'] = (log['outcome_payment'] == 'partially_paid').astype(int)
    log['outcome_unpaid'] = (log['outcome_payment'] == 'unpaid').astype(int)

    return log
