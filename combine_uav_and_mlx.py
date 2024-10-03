import pandas as pd

pi1_data = pd.read_excel('mlx_files/temperature_data_pi1.xlsx')
pi2_data = pd.read_excel('mlx_files/temperature_data_pi2.xlsx')
pi3_data = pd.read_excel('mlx_files/temperature_data_pi3.xlsx')
pi4_data = pd.read_excel('mlx_files/temperature_data_pi4.xlsx')
pi5_data = pd.read_excel('mlx_files/temperature_data_pi5.xlsx')
pi6_data = pd.read_excel('mlx_files/temperature_data_pi6.xlsx')
pi7_data = pd.read_excel('mlx_files/temperature_data_pi7.xlsx')
pi8_data = pd.read_excel('mlx_files/temperature_data_pi8.xlsx')

uav_data = pd.read_excel('uav_data_combined_mean.xlsx')
uav_mlx_combined_excel = 'uav_mlx_combined.xlsx'

combined_df = pd.DataFrame()


def get_mlx_temp_rows(capture_date):
    formatted_date = capture_date.strftime('%Y-%m-%d')

    match cam_no:
        case 1:
            return pi1_data.loc[(pi1_data['date'] == formatted_date)]
        case 2:
            return pi2_data.loc[(pi2_data['date'] == formatted_date)]
        case 3:
            return pi3_data.loc[(pi3_data['date'] == formatted_date)]
        case 4:
            return pi4_data.loc[(pi4_data['date'] == formatted_date)]
        case 5:
            return pi5_data.loc[(pi5_data['date'] == formatted_date)]
        case 6:
            return pi6_data.loc[(pi6_data['date'] == formatted_date)]
        case 7:
            return pi7_data.loc[(pi7_data['date'] == formatted_date)]
        case 8:
            return pi8_data.loc[(pi8_data['date'] == formatted_date)]


for index, row in uav_data.iterrows():
    date = row['date']
    cam_no = row['cam_no']
    day = row['day']
    mean = row['mean']
    median = row['median']
    percent95 = row['percent95']

    filtered_df = get_mlx_temp_rows(date)
    if len(filtered_df) > 0:

        # Extract the values from the value column
        temp_overall_values = filtered_df['mean_temp_overall']
        temp_canopy_values = filtered_df['mean_temp_canopy']

        print(temp_overall_values)
        print(temp_canopy_values)

        # Check if all values are 'N/A'
        if all(pd.isnull(temp_overall_values)):
            mlx_mean_temp_overall = 'N/A'
        else:
            # Remove 'N/A' values and calculate the mean
            temp_overall_valid_values = temp_overall_values[~pd.isnull(temp_overall_values)]
            mlx_mean_temp_overall = temp_overall_valid_values.mean()

        if all(pd.isnull(temp_canopy_values)):
            mlx_mean_temp_canopy = 'N/A'
        else:
            # Remove 'N/A' values and calculate the mean
            temp_canopy_valid_values = temp_canopy_values[~pd.isnull(temp_canopy_values)]
            mlx_mean_temp_canopy = temp_canopy_valid_values.mean()

    else:
        mlx_mean_temp_overall, mlx_mean_temp_canopy = 'N/A', 'N/A'
    row_id = index + 1
    data = {'index': row_id, 'day': day, 'cam_no': cam_no, 'date': date, 'uav_mean': mean,
            'uav_median': median, 'uav_percent95': percent95, 'mlx_overall_mean': mlx_mean_temp_overall,
            'mlx_mean_temp_canopy': mlx_mean_temp_canopy}
    combined_df = pd.concat([combined_df, pd.DataFrame(data, index=[index])], ignore_index=True)
    print(str(index + 1) + "===============")


combined_df.to_excel(uav_mlx_combined_excel, index=False)

