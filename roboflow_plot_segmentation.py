from roboflow import Roboflow
rf = Roboflow(api_key="8hdIUAZJ7iiIDsFr0FKW")
project = rf.workspace().project("mlx_thermal_img_segmentation")
model = project.version(1).model

# infer on a local image
print(model.predict("mlx_images/pi6_106_pic_2024-07-12_10-00-23.jpg", confidence=40).json())

# visualize your prediction
model.predict("mlx_images/pi6_106_pic_2024-07-12_10-00-23.jpg", confidence=40).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())