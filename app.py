import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"

def predict(image):
    with open(image, "rb") as f:
        files = {"file": f}
        response = requests.post(API_URL, files=files)

    result = response.json()

    return (
        result["class_name"],
        f"{result['confidence']:.2%}"
    )

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence")
    ],
    title="Finger Sign Classifier",
    description="Upload an image of a hand sign."
)

demo.launch()