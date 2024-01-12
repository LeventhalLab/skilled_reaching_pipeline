import os
import navigation_utilities
import pandas as pd
import utils
import numpy as np
from datetime import datetime


def get_trialdf_row(vid_metadata, trials_df):

    trial_dates = utils.datetime64_to_date_array(trials_df['session_date'].values)
    trialdate_df = trials_df.loc[trial_dates == vid_metadata['triggertime'].date()]
    session_df = trialdate_df[trialdate_df['date_session_num'] == vid_metadata['session_num']]
    trial_df = session_df[session_df['vid_number_in_name'] == vid_metadata['video_number']]

    return trial_df


def get_vidtrigger_ts(vid_metadata, trials_df):

    trial_df = get_trialdf_row(vid_metadata, trials_df)

    if trial_df.empty:
        # most likely, this was a trial that was performed after the photometry recording ended
        return None, None

    vidtrigger_ts = trial_df['vidtrigger_ts'].values[0]
    vidtrigger_interval = trial_df['vidtrigger_interval'].values[0]

    return vidtrigger_ts, vidtrigger_interval
