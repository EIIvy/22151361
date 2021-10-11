import torch
import torchvision.transforms
from PIL import Image
from torch import nn

image_path = "./datasets/dog.jpg"
image1_path = "./datasets/airplane.png"
image2_path = "./datasets/cat.png"
image = Image.open(image_path)
image1 = Image.open(image1_path)
image2 = Image.open(image2_path)
# print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
image = transform(image)
image1 = transform(image1)
image2 = transform(image2)
# print(image.shape)

#class  {0:'airplane' 1:'automobile' 2:'bird' 3:'cat' 4:'deer' 5:'dog' 6:'frog' 7:'horse' 8:'ship' 9:'truck'}
# 加载网络模型
model = torch.load("../network/network_75.pth", map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
image1 = torch.reshape(image1, (1, 3, 32, 32))
image2 = torch.reshape(image2, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
    output1 = model(image1)
    output2 = model(image2)
print(output)
print(output1)
print(output2)
print(output.argmax(1))
print(output1.argmax(1))
print(output2.argmax(1))
