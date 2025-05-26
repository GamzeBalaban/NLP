from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import uvicorn

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# FastAPI initialization
app = FastAPI()

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model path
MODEL_PATH = "bert_model2.pt" 

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load the trained model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Move model to device
model.to(device)
model.eval()

# API input model
class TextInput(BaseModel):
    text: str


labels = ["Positive", "Neutral", "Negative"] 
def clean_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [t for t in words if t not in stop_words and t not in punctuations and t.isalpha()]
    pos_tags = pos_tag(words)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(lemmatized)


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
    uvicorn.run(app, host="0.0.0.0", port=8000)
