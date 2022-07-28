# """
# Handles inference using a model.
# See https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
# """

# import io
# import json
# import torchvision.transforms as transforms
# from PIL import Image

# # Import your model
# from torchvision import models

# imagenet_class_index = json.load(open('imagenet_class_index.json'))


# # Make sure to pass `pretrained` as `True` to use the pretrained weights:
# model = models.resnet18(pretrained=True)
# # Since we are using our model only for inference, switch to `eval` mode:
# model.eval()


# # Takes image data in bytes, applies the series of transforms and returns a tensor.
# def transform_image(image_bytes):
# 	my_transforms = transforms.Compose([transforms.Resize(255),
# 										transforms.CenterCrop(224),
# 										transforms.ToTensor(),
# 										transforms.Normalize(
# 											[0.485, 0.456, 0.406],
# 											[0.229, 0.224, 0.225])])
# 	image = Image.open(io.BytesIO(image_bytes))
# 	return my_transforms(image).unsqueeze(0)


# def get_prediction(filename):
# 	with open(filename, 'rb') as f:
# 		image_bytes = f.read()
# 		tensor = transform_image(image_bytes=image_bytes)
# 	outputs = model.forward(tensor)
# 	# The tensor y_hat will contain the index of the predicted class id.
# 	# However, we need a human readable class name. For that we need a class 
# 	# id to name mapping.
# 	_, y_hat = outputs.max(1)
# 	predicted_idx = str(y_hat.item())
# 	return imagenet_class_index[predicted_idx]

# 	return "Unsure", -1