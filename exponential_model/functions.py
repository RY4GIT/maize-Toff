from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def exp_model(t, delta_theta, theta_w, tau):
    return delta_theta * np.exp(-t/tau) + theta_w


def calc_dSdt(cropmodel_output, precip_thresh, dSdt_positive_thresh, dSdt_noise_thresh, plot_results=False):
    """
    Function to detect drydown and calculate dS/dt
    [input]
    cropmodel_output: dataframe of cropmodel output
    timestep: 
    [output]
    drydown_events: dataframe of cropmodel output, with calculated ds/dt & event-wise list 
    """

    ############# Calculate dSdt ###############

    precip_mask = cropmodel_output['R'].where(cropmodel_output['R'] < precip_thresh)
    no_sm_record_but_precip_present =  cropmodel_output['R'].where((precip_mask.isnull()) & (cropmodel_output['s'].isnull()))

    # Allow detecting soil moisture increment even if there is no SM data in between before/after rainfall event
    # NaN data is allowed up to 10 days
    cropmodel_output['sm_for_dS_calc'] = cropmodel_output['s'].ffill() 

    # Calculate dS
    cropmodel_output['dS'] = cropmodel_output['sm_for_dS_calc'].bfill(limit=5).diff().where(cropmodel_output['sm_for_dS_calc'].notnull().shift(periods=+1))

    # Drop the dS where  (precipitation is present) && (soil moisture record does not exist)
    cropmodel_output['dS'] = cropmodel_output['dS'].where((cropmodel_output['dS'] > -1) & (cropmodel_output['dS'] < 1))

    # Calculate dt
    non_nulls = cropmodel_output['sm_for_dS_calc'].isnull().cumsum()
    nan_length = non_nulls.where(cropmodel_output['sm_for_dS_calc'].notnull()).bfill()+1 - non_nulls +1
    cropmodel_output['dt'] = nan_length.where(cropmodel_output['sm_for_dS_calc'].isnull()).fillna(1)

    # Calculate dS/dt
    cropmodel_output['dSdt'] = cropmodel_output['dS']/cropmodel_output['dt']
    cropmodel_output['dSdt'] = cropmodel_output['dSdt'].shift(periods=-1)

    cropmodel_output.loc[cropmodel_output['s'].shift(-1).isna(), 'dSdt'] = np.nan

    ############# Define drydown events ###############

    negative_increments = cropmodel_output.dSdt < 0
    positive_increments = cropmodel_output.dSdt > dSdt_positive_thresh
    # To avoid noise creating spurious drydowns, identified drydowns were excluded from the analysis when the positive increment preceding the drydown was less than two times the target unbiased root-mean-square difference for SMAP observations (0.08).

    # Negative dSdt preceded with positive dSdt
    cropmodel_output['event_start'] = negative_increments.values & np.concatenate(([False], positive_increments[:-1]))
    cropmodel_output['event_start'][cropmodel_output['event_start']].index

    event_end = np.zeros(cropmodel_output.shape[0], dtype=bool)
    # cropmodel_output['dS'] = cropmodel_output['dS'].shift(-1)
    for i in range(1, cropmodel_output.shape[0]):
        if cropmodel_output['event_start'][i]:
            start_index = i
            for j in range(i+1, cropmodel_output.shape[0]):
                if np.isnan(cropmodel_output['dS'][j]):
                    None
                if cropmodel_output['dS'][j] >= dSdt_noise_thresh or cropmodel_output['R'][j] > precip_thresh:
                    # Any positive increment smaller than 5% of the observed range of soil moisture at the site is excluded (if there is not precipitation) if it would otherwise truncate a drydown. 
                    event_end[j] = True
                    break

    # create a new column for event_end
    cropmodel_output['event_end'] = event_end
    cropmodel_output['event_end'] = cropmodel_output['event_end'].shift(-1)
    cropmodel_output = cropmodel_output[:-1]

    ############# Plot results ###############

    if plot_results: 
        fig, (ax11, ax12) = plt.subplots(2,1, figsize=(20, 5))
        cropmodel_output.s.plot(ax=ax11, alpha=0.5)
        ax11.scatter(cropmodel_output.s[cropmodel_output['event_start']].index, cropmodel_output.s[cropmodel_output['event_start']].values, color='orange', alpha=0.5)
        ax11.scatter(cropmodel_output.s[cropmodel_output['event_end']].index, cropmodel_output.s[cropmodel_output['event_end']].values, color='orange', marker='x', alpha=0.5)
        cropmodel_output.R.plot(ax=ax12, alpha=0.5)
        # fig.savefig(os.path.join(output_path2, f'{target_station}_timeseries.png'))

    ############# Preparing output dataframe ###############
    start_indices = cropmodel_output[cropmodel_output['event_start']].index
    end_indices = cropmodel_output[cropmodel_output['event_end']].index
    cropmodel_output['dSdt(t-1)'] = cropmodel_output.dSdt.shift(+1)

    # Create a new DataFrame with each drydown_event containing a list of soil moisture values between each pair of event_start and event_end
    event_data = [{'event_start': start_index, 
                'event_end': end_index, 
                'soil_moisture': list(cropmodel_output.loc[start_index:end_index, 's'].values),
                'precip': list(cropmodel_output.loc[start_index:end_index, 'R'].values),
                'delta_theta': cropmodel_output.loc[start_index, 'dSdt(t-1)'],
                'LAI': list(cropmodel_output.loc[start_index:end_index, 'LAI'].values),
                'ET': list(cropmodel_output.loc[start_index:end_index, 'ET'].values),
                } 
                for start_index, end_index in zip(start_indices, end_indices)]
    event_df = pd.DataFrame(event_data)

    # Only retain events more than 4 days
    drydown_events = event_df[event_df['soil_moisture'].apply(lambda x: pd.notna(x).sum()) >= 4].copy()
    drydown_events = drydown_events.reset_index(drop=True)

    return drydown_events



def fit_exp_model(drydown_event, min_sm_values_at_the_pt=0.03):
    """
    Function to fit exponential 
    [input]
    drydown_events: dataframe of cropmodel output, with calculated ds/dt & event-wise list 
    [output]
    exp_fit_params: dataframe of fitted parameters
    """

    start_date = drydown_event['event_start']
    end_date = drydown_event['event_end']
    delta_theta = drydown_event['delta_theta']
    soil_moisture_subset = np.asarray(drydown_event['soil_moisture'])

    t = np.arange(0, len(soil_moisture_subset),1)
    soil_moisture_range = np.nanmax(soil_moisture_subset) - np.nanmin(soil_moisture_subset)
    soil_moisture_subset_min = np.nanmin(soil_moisture_subset)
    soil_moisture_subset_max = np.nanmax(soil_moisture_subset)
    x = t[~np.isnan(soil_moisture_subset)]
    y = soil_moisture_subset[~np.isnan(soil_moisture_subset)]
    
    # exp_model(t, delta_theta, theta_w, tau):
    # bounds  = [(0, min_sm_values_at_the_pt, 0), (np.inf, soil_moisture_subset_min, np.inf)]
    # p0      = [0.5*soil_moisture_range, (soil_moisture_subset_min+min_sm_values_at_the_pt)/2, 1]
    bounds  = [(0, min_sm_values_at_the_pt-0.01, 0), (np.inf, min_sm_values_at_the_pt+0.01, np.inf)]
    p0      = [0.5*soil_moisture_range, min_sm_values_at_the_pt, 1]

    try: 
        popt, pcov = curve_fit(f=exp_model, xdata=x, ydata=y, p0=p0, bounds=bounds)
        # popt: Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized
        # pcov: The estimated covariance of popt
        y_opt = exp_model(x, *popt)
        residuals = y - y_opt
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.nanmean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        exp_fit_params = {'event_start': start_date, 'event_end': end_date, 'delta_theta': popt[0], 'theta_w': popt[1], 'tau': popt[2], 'r_squared': r_squared, 'opt_drydown': y_opt.tolist()}
    except:
        print('Error raised')
    return exp_fit_params



def plot_expfit_results(i, drydown_event):
    # Plot each drydown_event of the event DataFrame as a time series
    # Determine the number of columns needed for the subplots grid
    # num_events = len(drydown_event)
    # num_cols = 3
    # num_rows = int(num_events / num_cols) + int(num_events % num_cols != 0)
    fig, axes = plt.subplots(figsize=(3,3))

    x = np.arange(drydown_event['event_start'], drydown_event['event_end']+1, 1)
    y = np.asarray(drydown_event['soil_moisture'])
    y_opt = np.asarray(drydown_event['opt_drydown'])
    r_squared = drydown_event['r_squared']
    tau = drydown_event['tau']
    theta_w = drydown_event['theta_w']
    delta_theta = drydown_event['delta_theta']
    # try:
    axes.scatter(x, y)
    axes.plot(x[~np.isnan(y)], y_opt, alpha=.7)
    axes.set_title(f'Event {i} (R2={r_squared:.2f}; tau={tau:.2f}\ntheta_w={theta_w:.2f}; delta_theta={delta_theta:.2f})', fontsize = 10)
    axes.set_xlabel('Date')
    axes.set_ylabel('Soil Moisture')
    axes.set_xlim([drydown_event['event_start'], drydown_event['event_end']])
    fig.tight_layout()
    fig.show()



