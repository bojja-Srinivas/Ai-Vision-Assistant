import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import pytesseract
import pyttsx3
import torch
import os
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\bojja\devil\tesseract.exe"

# Load environment variables (assuming a file named "Google api key.txt" with your key)
from dotenv import load_dotenv
load_dotenv()
f = open("Google api key.txt")
key = f.read()
# Configure your Gemini API key
genai.configure(api_key=key)

# Streamlit App Configuration
st.set_page_config(page_title="AISight assistant", layout="centered")

# Function to get response from Gemini model with image data
def get_response(input_prompt, image_data):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

# Function to convert uploaded image to byte data
def image_to_bytes(uploaded_file):
    try:
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]

        return image_parts
    
    except Exception as e:
        raise FileNotFoundError(f"Failed to process image. Please try again. Error: {e}")

# Function to extract text from image using Tesseract OCR
def extract_text_from_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)

        # pytesseract to extract text
        extracted_text = pytesseract.image_to_string(img)

        if not extracted_text.strip():
            return "No text found in the image."
        
        return extracted_text

    except Exception as e:
        raise ValueError(f"Failed to extract text. Error: {e}")

# Function for text-to-speech conversion using pyttsx3
def text_to_speech_pyttsx3(text):
    try:
        # Initialize text-to-speech engine
        engine = pyttsx3.init()  
        engine.stop()  
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
        engine.stop()  # Stop the engine after use
        
    except Exception as e:
        raise RuntimeError(f"Failed to convert text to speech. Error: {e}")

# Load object detection model (Faster R-CNN) - This is cached for efficiency
@st.cache_resource
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

object_detection_model = load_object_detection_model()

def detect_objects(image, threshold=0.5, iou_threshold=0.5):
    try:
        # Transform image to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image)
        
        # Get predictions
        predictions = object_detection_model([img_tensor])[0]
        
        # Perform Non-Maximum Suppression
        keep = torch.ops.torchvision.nms(predictions['boxes'], predictions['scores'], iou_threshold)
        
        # Filter results based on NMS and score threshold
        filtered_predictions = {
            'boxes': predictions['boxes'][keep],
            'labels': predictions['labels'][keep],
            'scores': predictions['scores'][keep]
        }
        
        return filtered_predictions
    except Exception as e:
        raise RuntimeError(f"Failed to detect objects. Error: {e}")

# COCO class labels (list of 80 categories)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Highlight detected objects in the image
def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    labels = predictions['labels']
    boxes = predictions['boxes']
    scores = predictions['scores']

    for label, box, score in zip(labels, boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[label.item()]  # Map label ID to class name
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{class_name} ({score:.2f})", fill="black")
    return image

# Response function for Personalized Assistance
def get_assistance_response(input_prompt, image_data):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

# Prompt Engineering
input_prompt = """
You are an AI assistant designed to assist visually impaired individuals by analyzing images and providing descriptive outputs.
Your Task is:
Analyze the Uploaded Image: Examine the content of the uploaded image and describe it in clear and simple language. This includes identifying and explaining the main elements within the image.
Provide Detailed Information: Offer comprehensive details about objects, people, settings, or activities present in the scene. This helps visually impaired users understand their surroundings better.
"""


st.title("AI Powered assistant for Visually Impaired Persons")
uploaded_file = st.file_uploader("*Upload an image:*", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None: 
    image = uploaded_file.read() 
    st.image(image, caption='Uploaded Image', use_column_width=True)

bt1, bt2, bt3, bt4 = st.columns(4)
scene_analysis_button = bt1.button("Real-Time Scene Understanding")
object_detection_button = bt2.button("Object Detection")
assist_button = bt3.button("Personalized Assistance")
text_tts_button = bt4.button("text Extraction")
stop_audio_button = st.button("Audio")
# Object Detection
if object_detection_button and uploaded_file:
    with st.spinner("Detecting objects..."):
        st.subheader("Detected Objects:")
        image = Image.open(uploaded_file)
        predictions = detect_objects(image)
        image_with_boxes = draw_boxes(image.copy(), predictions)
        st.image(image_with_boxes, caption="Objects Highlighted", use_column_width=True)

# Personalized Assistance
if assist_button and uploaded_file:
    with st.spinner("Providing uploaded task-specific assistance..."):
        st.subheader("Assistance:")
        image_data = image_to_bytes(uploaded_file)
        assist_prompt = """
        You are a helpful AI assistant. Analyze the uploaded image and identify tasks
        you can assist with, such as recognizing objects or reading labels.
        """
        response = get_assistance_response(assist_prompt, image_data)
        st.write(response)
         # Convert response to audio

        text_to_speech_pyttsx3(response)

# Scene Analysis
if scene_analysis_button and uploaded_file:
    with st.spinner("Analyzing Uploaded Image..."):
        st.subheader("Scene Description:")
        image_data = image_to_bytes(uploaded_file)
        response = get_response(input_prompt, image_data)
        st.write(response)
        text_to_speech_pyttsx3(response)

if text_tts_button and uploaded_file:
    with st.spinner("Extracting text from uploaded image..."):
        text = extract_text_from_image(uploaded_file)
        st.write(text)
        if text.strip():
            text_to_speech_pyttsx3(text)
if stop_audio_button:  
    try:  
        engine = pyttsx3.init()  
        engine.stop()  
        audio_playing = False  
    except Exception as e:  
        st.error(f"Failed to stop the audio. Error: {e}") 
