from PIL import Image

import cv2
import pandas as pd
import numpy as np
import string
import cmapy
from datetime import datetime

# specify the cam number
cam_number = '7'

process_thermal = True
mlx_temp_mappings = True

image_folder = 'C:/Users/lochana.marasingha/Desktop/azure_mount_othello_2024_after_pre-check/pi' + cam_number + '/'

f = open('files/CombinedData' + cam_number + '.csv', "r")
text = f.read()
_colormap_list = ['jet', 'bwr', 'seismic', 'coolwarm', 'PiYG_r', 'tab10', 'tab20', 'gnuplot2', 'brg']
_colormap_index = 0
output_folder = 'mlx_images/'
mlx_generated_image_prefix = 'pi' + cam_number
split_text = text.split("\n\n")
print("No of captures: ", len(split_text) - 1)
pi_camera_counter = 0
lepton_counter = 0
mlx_counter = 0

if process_thermal:
    if mlx_temp_mappings:
        excel_writer = pd.ExcelWriter('mlx_files/temp_mappings_pi' + cam_number + '.xlsx')
    temperature_data_excel = 'mlx_files/temperature_data_pi' + cam_number + '.xlsx'

df = pd.DataFrame()


def ktof(val):
    return (1.8 * ktoc(val) + 32.0)


def ktoc(val):
    return (val - 273.15) / 100.0


def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)


def display_temperature(img, val_k, loc, color):
    val = ktof(val_k)
    cv2.putText(img, "{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)


def map_to_excel_sheets(temp_values):
    print("MLX temperature no of values: ", len(temp_values))

    mlx_temp_data = np.array(temp_values)  # Replace with your actual data

    # Reshape the array to 24x32
    reshaped_mlx_data = mlx_temp_data.reshape(24, 32)

    # Create a pandas DataFrame from the reshaped array
    mlx_dataframe = pd.DataFrame(reshaped_mlx_data)
    sheet_name = mlx_generated_image_prefix + '_' + str(mlx_counter) + '_' + \
                 mlx_camera_img_location.split("/")[-1].split(".")[0]
    mlx_dataframe.to_excel(excel_writer, sheet_name=sheet_name)
    for letter in string.ascii_uppercase:
        excel_writer.sheets[sheet_name].column_dimensions[letter].width = 5
    excel_writer.sheets[sheet_name].column_dimensions['AA'].width = 5
    excel_writer.sheets[sheet_name].column_dimensions['AB'].width = 5
    excel_writer.sheets[sheet_name].column_dimensions['AC'].width = 5
    excel_writer.sheets[sheet_name].column_dimensions['AD'].width = 5
    excel_writer.sheets[sheet_name].column_dimensions['AF'].width = 5
    excel_writer.sheets[sheet_name].column_dimensions['AG'].width = 5

    # generate image without interpolation
    temp_min = np.min(reshaped_mlx_data)
    temp_max = np.max(reshaped_mlx_data)
    rescaled_image = np.nan_to_num(reshaped_mlx_data)
    norm = np.uint8((rescaled_image - temp_min) * 255 / (temp_max - temp_min))
    norm.shape = (24, 32)

    image = cv2.applyColorMap(norm, cmapy.cmap(_colormap_list[_colormap_index]))

    file_name = output_folder + sheet_name + '.jpg'
    cv2.imwrite(file_name, image)


for i in range(len(split_text) - 1):
    observation = split_text[i]
    print("Observation: ", i + 1)
    observation_split = observation.split(",")
    pi_camera_img_location = observation_split[0]
    lepton_camera_img_location = observation_split[1]
    mlx_camera_img_location = observation_split[2]
    if pi_camera_img_location != "N/A": pi_camera_counter += 1
    if lepton_camera_img_location != "N/A": lepton_counter += 1
    if mlx_camera_img_location != "N/A": mlx_counter += 1

    mlx_temp_C = observation_split[3]
    mlx_temp_F = observation_split[4]
    mlx_temp_values = observation_split[5]
    captured_time = float(observation_split[6])

    print("Pi camera image location: ", pi_camera_img_location)
    print("Lepton camera image location: ", lepton_camera_img_location)
    print("MLX camera location: ", mlx_camera_img_location)
    print("MLX temperature celcius: ", mlx_temp_C)
    print("MLX temperature Fahrenheit: ", mlx_temp_F)
    print("Images captured timestamp: ", captured_time)

    # ============================================= Start MLX Thermal Sensor Processing ================================

    if process_thermal:
        modified_temp_values = mlx_temp_values[mlx_temp_values.index('[') + 1:mlx_temp_values.index(']')]

        temp_value_rows = modified_temp_values.split("\n")

        temp_values_str = []
        for j in range(len(temp_value_rows)):
            temp_values_str = temp_values_str + temp_value_rows[j].split()

        all_temp_values = [float(str_value) for str_value in temp_values_str]

        if len(all_temp_values) == 768:
            percentile_85 = np.percentile(all_temp_values, 85)

            filtered_temp_values = [i for i in all_temp_values if i <= percentile_85]
            average_canopy_temperature = sum(filtered_temp_values) / len(filtered_temp_values)
            mlx_temp_C = float(mlx_temp_C)
            if mlx_temp_mappings:
                map_to_excel_sheets(all_temp_values)
        else:
            average_canopy_temperature = 'N/A'
        # =IFS(1713337200000 < B2 < 1713423600000, "2024/04/17", 1715238000000 < B2 < 1715324400000, "2024/05/09",
        #      1716447600000 < B2 < 1716534000000, "2024/05/23", 1717052400000 < B2 < 1717138800000, "2024/05/30",
        #      1717657200000 < B2 < 1717743600000, "2024/06/06", 1718262000000 < B2 < 1718348400000, "2024/06/13",
        #      1718953200000 < B2 < 1719039600000, "2024/06/21", 1719903600000 < B2 < 1719990000000, "2024/07/02")

        dt = datetime.fromtimestamp(captured_time / 1000.0)
        date = dt.strftime("%Y-%m-%d")
        data = {'index': i + 1, 'timestamp': captured_time, 'date': date, 'lepton_img_loc': lepton_camera_img_location,
                'mlx_img_loc': mlx_camera_img_location, 'mean_temp_overall': mlx_temp_C,
                'mean_temp_canopy': average_canopy_temperature}
        df = pd.concat([df, pd.DataFrame(data, index=[i])], ignore_index=True)

    # ============================================= End MLX Thermal Sensor Processing ==================================

    # =========================================== Start Lepton Thermal Camera Processing =======+=======================

    if lepton_camera_img_location != 'N/A':
        lepton_img_name = lepton_camera_img_location.split("/")[-1]
        lepton_image = image_folder + 'lepton/' + lepton_img_name
        print("Lepton Image: " + lepton_image)
        read_lepton_image = cv2.imread(lepton_image)

        # image = Image.open('C:/Users/lochana.marasingha/Downloads/date_4-4-2024_16.28.19_1.png')
        # # Get the mode (which indicates the bit depth)
        # mode = image.mode
        # print("Mode:", mode)

        try:
            lepton_frame_y16 = cv2.cvtColor(read_lepton_image, cv2.COLOR_BGR2GRAY)

            # # Convert to YUV
            # img_yuv = cv2.cvtColor(read_lepton_image, cv2.COLOR_BGR2YUV)
            # # Extract Y channel
            # y_channel = img_yuv[:, :, 0]
            # Scale to 16-bit
            print(lepton_frame_y16)

            grayscale_16bit = np.uint16(lepton_frame_y16 * (65535 / 255))
            print(grayscale_16bit)

            # frame_normalized = cv2.normalize(grayscale_16bit, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
            # print(frame_normalized)
            # colored_image = cv2.applyColorMap(frame_normalized, cv2.COLORMAP_JET)
            # cv2.imwrite('converted_lepton_images/' + lepton_img_name, grayscale_16bit)
            # cv2.imwrite('uvc_radiometry_read_images/' + lepton_img_name, read_lepton_image)
            degree_frame = grayscale_16bit / 100 - 273.15
            # degree_frame = ktoc(frame_normalized)  # Convert to Celsius
            mean = np.mean(degree_frame)
            print(mean)

        except Exception as e:
            print("Error converting to Y16:", e)
    # =========================================== End Lepton Thermal Camera Processing =======+=======================

if process_thermal:
    df.to_excel(temperature_data_excel, index=False)
    if mlx_temp_mappings:
        excel_writer.close()
print("Pi Camera Counter: ", pi_camera_counter)
print("Lepton Camera Counter: ", lepton_counter)
print("MLX Camera Counter: ", mlx_counter)
cv2.destroyAllWindows()
f.close()
