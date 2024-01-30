import pandas as pd
import navigation_utilities
import skilled_reaching_io
from datetime import datetime

def create_empty_skilledreaching_dataframe():
    srdf_variables = {'trial_num': pd.Series(dtype='int'),
                      'overall_session_num': pd.Series(dtype='int'),
                      'session_date': pd.to_datetime(1),
                      'session_type': pd.Series(dtype='str'),
                      'date_session_num': pd.Series(dtype='int'),
                      'trial_num_in_session': pd.Series(dtype='int8'),
                      'vid_number_in_session': pd.Series(dtype='int8'),
                      'vid_number_in_name': pd.Series(dtype='int8'),
                      'act3_ts': pd.Series(dtype='float'),
                      'vidtrigger_ts': pd.Series(dtype='float'),
                      'outcome': pd.Series(dtype='int8'),
                      'act3_interval': pd.Series(dtype='int8'),
                      'vidtrigger_interval': pd.Series(dtype='int8'),  # this is the valid signal interval in which this vidtrigger event occurred (-1 if it was outside a valid recording interval)
                      'session_duration': pd.Series(dtype='float')
                      }
    sr_df = pd.DataFrame(data=srdf_variables, index=[])

    return sr_df


def create_rat_srdf(ratID, parent_directories, expt_name):
    '''
    0 – No pellet, mechanical failure
    1 -  First trial success (obtained pellet on initial limb advance). If more than one pellet on pedestal, successfully grabbing any pellet counts as success for scores 1 and 2
    2 -  Success (obtain pellet, but not on first attempt)
    3 -  Forelimb advance - pellet dropped in box
    4 -  Forelimb advance - pellet knocked off shelf
    5 -  Obtained pellet with tongue
    6 -  Walked away without forelimb advance, no forelimb advance
    7 -  Reached, pellet remains on shelf
    8 - Used only contralateral paw
    9 – Laser/video fired at the wrong time
    10 – Used preferred paw after obtaining or moving pellet with tongue
    11 – Obtained pellet with preferred paw after using non-preferred paw
    '''

    scores_xlsx = navigation_utilities.find_scores_xlsx(parent_directories, expt_name)
    scores_df = skilled_reaching_io.read_xlsx_scores(scores_xlsx, ratID)

    rat_srdf = create_empty_skilledreaching_dataframe()
    # get all column headers in the scores_df dataframe that include the ratID. Column headers should be "vid_number"
    # for the first column, then ratID_YYYYMMDD_xx where xx is 01, 02, etc. for each column with actual scores
    session_cols = [col for col in scores_df.columns if ratID in col]
    if scores_df.empty:
        # if this rat doesn't have any scores yet, just return an empty dataframe
        return rat_srdf

    vid_numbers = scores_df['vid_number']

    num_rows = scores_df.shape[0]
    trial_num = 0
    for i_session, session in enumerate(session_cols):
        session_scores = scores_df[session]
        session_nameparts = session.split('_')
        session_date = datetime.strptime(session_nameparts[1], '%Y%m%d')
        session_type = ''   # todo: figure out whether this is a chrimson session

        date_session_num = int(session_nameparts[2])

        trial_num_in_session = 0
        vid_number_in_session = 0
        for i_row in range(num_rows):
            if np.isnan(session_scores[i_row]):
                # no video was recorded for this video number
                continue
            vid_number_in_session += 1   # important to make sure we keep track of which timestamp goes with which video
            if session_scores[i_row] in (6, 9):
                # there was a video recorded but no reach; add a row so we can keep track, but don't count it as a reach
                new_row_dict = {'trial_num': np.nan,
                               'overall_session_num': i_session,
                               'session_date': session_date,
                               'session_type': session_type,
                               'date_session_num': date_session_num,
                               'trial_num_in_session': np.nan,
                               'vid_number_in_session': vid_number_in_session,
                               'vid_number_in_name': vid_numbers[i_row],
                               'act3_ts': np.nan,
                               'vidtrigger_ts': np.nan,
                               'outcome': session_scores[i_row],
                               'act3_interval': -2,
                               'vidtrigger_interval': -2,  # this is the valid signal interval in which this vidtrigger event occurred (-1 if it was outside a valid recording interval, -2 if not a valid video)
                               'session_duration': 0.
                                }
            else:
                trial_num += 1
                trial_num_in_session += 1
                new_row_dict = {'trial_num': trial_num,
                               'overall_session_num': i_session,
                               'session_date': session_date,
                               'session_type': session_type,
                               'date_session_num': date_session_num,
                               'trial_num_in_session': trial_num_in_session,
                               'vid_number_in_session': vid_number_in_session,
                               'vid_number_in_name': vid_numbers[i_row],
                               'act3_ts': np.nan,
                               'vidtrigger_ts': np.nan,
                               'outcome': session_scores[i_row],
                               'act3_interval': -3,
                               'vidtrigger_interval': -3,  # this is the valid signal interval in which this vidtrigger event occurred (-1 if it was outside a valid recording interval); placeholder because it hasn't been assigned yet
                               'session_duration': 0.
                                }

            new_row = pd.DataFrame(data=new_row_dict, index=[0])
            rat_srdf = pd.concat([rat_srdf, new_row], ignore_index=True)

    return rat_srdf