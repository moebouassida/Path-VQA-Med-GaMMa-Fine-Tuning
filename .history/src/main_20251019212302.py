from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import requests
from io import BytesIO
from unsloth import FastVisionModel

app = FastAPI(title="Path-VQA Med-GaMMa Inference API")

class QARequest(BaseModel):
    image_url: str
    question: str

# Load fine-tuned model (from outputs/)
model, processor = FastVisionModel.from_pretrained("outputs", load_in_4bit=True)
model.eval()

@app.post("/predict/")
def predict_answer(req: QARequest):
    # Download image from URL
    response = requests.get(req.image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": req.question},
                {"type": "image", "image": img}
            ]
        }
    ]

    # Generate answer
    outputs = model.generate(conversation, max_new_tokens=256)
    answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer}
