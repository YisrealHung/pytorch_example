import torch
import numpy as np
from PIL import Image
from torchvision import transforms,models
import cv2


model = models.resnet18(pretrained = True)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('final_model.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

label = np.array(['face', 'mask'])

# 數據預處理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # BGR
    ret, frame = cap.read()
    # RGB
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    outputs = model(img)
    _, predicted = torch.max(outputs,1)
    print(label[predicted.item()])

    cv2.putText(frame, label[predicted.item()], (10, 40), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    keyin = cv2.waitKey(1) & 0xFF
    if keyin == ord('q'):
        break
