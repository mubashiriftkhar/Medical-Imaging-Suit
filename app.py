
from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware

import torch
from torchvision.transforms import transforms
from PIL import Image
import io

model=torch.jit.load('BrainTumorModel1.pt',map_location=torch.device("cpu"))
model.eval()



transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.485, 0.485],
        std=[0.229, 0.229, 0.229]
    )

])


app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def Process(file:UploadFile=File(...)):
    imageBytes=await file.read()
    image=Image.open(io.BytesIO(imageBytes))
    image=transform(image).unsqueeze(0)
    with torch.no_grad():
         output=model(image)
         _,predicted=torch.max(output,1)
         className=[0,1][predicted.item()]
         if className==0:
              return "Model Detected Cancer in Brest."
         else:
              return "Model does not Detected any Cancer in Brest."
