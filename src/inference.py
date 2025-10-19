from unsloth import FastVisionModel
from PIL import Image
import requests
from io import BytesIO

# Load model
model, processor = FastVisionModel.from_pretrained("outputs", load_in_4bit=True)
model.eval()

def predict(image_url: str, question: str):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image", "image": img}]}
    ]

    outputs = model.generate(conversation, max_new_tokens=256)
    answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    image_url = "https://example.com/pathology_image.jpg"
    question = "What type of tissue is shown?"
    answer = predict(image_url, question)
    print("Answer:", answer)
