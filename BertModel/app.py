from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# FastAPI initialization
app = FastAPI()

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model path
MODEL_PATH = "bert_model2.pt"  # Kaydettiğin dosya adını doğru yazdım

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize the model architecture first
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load the trained model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Move model to device
model.to(device)
model.eval()

# API input model
class TextInput(BaseModel):
    text: str

# Label list (senin görevine göre burayı değiştirdim)
labels = ["Positive", "Neutral", "Negative"]  # Örneğin senin sınıfların bunlarsa böyle

# API endpoint
@app.post("/predict")
async def predict_sentiment(input_data: TextInput):
    # Tokenize input data
    tokens = tokenizer(input_data.text, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Run the model
        output = model(**tokens)

        # Get prediction
        prediction = torch.argmax(output.logits, dim=1).item()
        predicted_label = labels[prediction]

    return {"text": input_data.text, "predicted_label": predicted_label}

# Run API (opsiyonel)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
