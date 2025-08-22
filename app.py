from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from torchvision.transforms import transforms
from PIL import Image
import io





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
# ---- Load all models at startup (one time) ----
models = {
    "BreastCancer": torch.jit.load("Breast_Cancer.pt", map_location="cpu"),
    "BrainTumor": torch.jit.load("BrainTumor.pt", map_location="cpu"),
    "Pneumonia": torch.jit.load("Pneumonia.pt", map_location="cpu"),
    "KidneyStone": torch.jit.load("KidneyStone.pt", map_location="cpu"),
}

# Set all models to eval mode
for model in models.values():
    model.eval()

# ---- Inference wrapper ----
def run_inference(model, image, labels):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        pred_class = labels[predicted.item()]
        return pred_class
labels_map = {
    "BreastCancer": ["No Cancer Detected", "Cancer Detected"],
    "BrainTumor": ['glioma','meningioma','notumor','pituitary'],
    "Pneumonia": ["No Pneumonia Detected", "Pneumonia Detected"],
    "KidneyStone": ["No Stone Detected", "Stone Detected"],
}
@app.post("/upload")
async def Process(file:UploadFile=File(...),Selection="BreastCancer"):
    imageBytes=await file.read()
    image=Image.open(io.BytesIO(imageBytes))
    image=transform(image).unsqBreastCancerueeze(0)
     # Look up the function dynamically
    if Selection in models:
        result = run_inference(models[Selection], image, labels_map[Selection])
    else:
        result = "Invalid selection."

    return {"result": result}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)