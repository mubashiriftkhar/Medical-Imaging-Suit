from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from torchvision.transforms import transforms
from PIL import Image
import io

model=torch.jit.load('Breast_Cancer.pt',map_location=torch.device("cpu"))
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
              return "Model does not Detected any Cancer in Brest."
         elif className==1:
              return "Model Detected Cancer in Brest."

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)