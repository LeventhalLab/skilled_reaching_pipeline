import os
import navigation_utilities
import pandas as pd

def get_vidtrigger_ts(vid_metadata, trials_df):

    session_row = trials_df.loc[trials_df['session_date'] == vid_metadata['triggertime'].date()]

    session_df = trials_df[(trials_df['session_date'] == vid_metadata['triggertime'].date()) & \
                            (trials_df['date_session_num'] == vid_metadata['session_num'])]

    # find the session row from the rat_df dataframe
    # session_row = rat_df[reaching_df['session_date'] == video_metadata['triggertime'].date()]
    video_metadata = {
        'ratID': '',
        'rat_num': 0,
        'session_name': '',
        'boxnum': 99,
        'triggertime': datetime(1,1,1),
        'video_number': 0,
        'video_type': '',
        'video_name': '',
        'im_size': (1024, 2040)
    }