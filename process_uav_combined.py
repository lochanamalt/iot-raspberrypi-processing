import pandas as pd

uav_combined = pd.read_excel('uav_data_combined.xlsx')
uav_combined_mean_excel = 'uav_data_combined_mean.xlsx'

# Create a new DataFrame to store the mean values
result_df = pd.DataFrame()
    # Iterate over every 12 rows of the DataFrame
for i in range(0, len(uav_combined), 12):
    date = uav_combined.iloc[i]['date']
    cam_no = uav_combined.iloc[i]['cam_no']
    day = uav_combined.iloc[i]['day']

    # Extract the current 12 rows
    chunk = uav_combined.iloc[i:i + 12]
    columns_to_calculate = ["Mean", "Median", "Perc95"]
    # Calculate the mean for the specified columns
    means = chunk[columns_to_calculate].mean()

    data = {'day': day, 'cam_no': cam_no, 'date': date, 'mean': means["Mean"],
            'median': means["Median"], 'percent95': means["Perc95"]}
    result_df = pd.concat([result_df, pd.DataFrame(data, index=[day-1])], ignore_index=True)
    # Append the mean values to the result DataFrame


result_df.to_excel(uav_combined_mean_excel, index=False)


