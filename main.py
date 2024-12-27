import os
import json
import streamlit as st
from groq import Groq

# GROQ INITIALIZATION
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
client = Groq()


# Variable INITIALIZATION

if 'chatStarted' not in st.session_state:
    st.session_state.chatStarted = False
    st.session_state.chat_history = []
    st.session_state.image = None
    st.session_state.turn = "user"
    st.session_state.all_chat_history = []
    st.session_state.refresh_required = False
    st.session_state.end_conversation = False

# Functions

def predict_uploaded_image(uploaded_file, model_path='C:/Users/Niloy/Desktop/ChatBot/PCOS_detection.pth'):
    import torch
    from torchvision import transforms
    from PIL import Image
    import torch.nn as nn

    # Define the model architecture
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, 12, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3, 3),
                nn.Conv2d(12, 15, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3, 3),
                nn.Conv2d(15, 10, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3, 3),
                nn.Flatten(),
                nn.Linear(810, 2),
            )
        
        def forward(self, xb):
            return self.network(xb)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Read the image from the uploaded file
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Predict the output
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, dim=1)

    # Map class index to label
    class_labels = ['Infected', 'Not Infected']
    return class_labels[predicted_class.item()]


# def CNN(image):
#     # Placeholder for the CNN function
#     # return "The system predicted: The Ultrasonograph is showing PCOS spots! Tell the user to contact health care immidiately!"  # Mock classification output
#     return "<NEGATIVE>"


def generate_response():
    messages = [
        {'role': 'system', 'content': """You are PCOS BOT, a specialised LLM trained to provide information about PCOS, its symptoms and how to take care.
         you are chatting with the user. This means most of the time your lines should be a sentence or two. However, you can provide details in list or table if the user want.
         At first, introduce yourself to the user. Never use emojis, unless explicitly asked to.
         You are pretrained with proper documents and resources, so be confident about your findings and reasoning. You response must be friendly, humble and polite.
         Ask about her age, symptoms and other necessary information that you need politely.

         You have access to the Ultrasound PCOS detector tool. This can take an image as input and provide correct feedback with 95% accuracy. 
         If user wants to upload an image, say '<<CLASSIFY>>' with no preamble. 
         The system will detect it from your response and allow the user to upload an image. Then the system will provide you the output of the tool.

         # Response from system:
        <Infected>: If system gives you this, then the user likely have PCOS. Instruct her to contact with health care professional
        <Not Infected>: If system gives you this, then the user does not have PCOS. Instruct her how to maintain healthy lifestyle.
        
        Once done chatting, you can end the conversation by saying nice things and take care.
        If user does not co-operate or says inappropriate word, or the conversation is done, then you can forcefully end the conversation by responsing '<<END>>' with no preamble. 
         """},
        *st.session_state.chat_history
    ]
    response = client.chat.completions.create(
        model = "llama-3.3-70b-specdec",
        messages = messages
    )
    return response.choices[0].message.content

def remove_commands(text):
    text = text.replace("<<END>>","")
    return text.replace("<<CLASSIFY>>","")

# Page Decoration 
st.set_page_config(
    page_title = "PCOS ChatBot",
    page_icon = "🩺",
    layout = "centered"
)
st.title("🩺 PCOS Chatbot")

# Start Screen
if not st.session_state.chatStarted:
    st.html("""<h1>Welcome to PCOS Chatbot!</h1>
    <p>Here are the things you can do:</p>
    <ol>
            <li><strong>Learn about PCOS:</strong> Understand what PCOS is and how it affects you.</li>
            <li><strong>Discover the symptoms:</strong> Get information on common symptoms of PCOS.</li>
            <li><strong>Prevention tips:</strong> Learn how to reduce the risk of PCOS.</li>
            <li><strong>Self-care guidance:</strong> Find out how to take care of yourself effectively.</li>
            <li><strong>Upload your ultrasound image:</strong> Use our core feature to detect PCOS</li>
    </ol>""")

if not st.session_state.end_conversation:
    user_prompt = st.chat_input("Ask me!")
    if user_prompt:
        st.session_state.chatStarted = True
        st.session_state.turn = "agent"
        st.session_state.chat_history.append({'role': 'user', 'content': user_prompt})
        st.session_state.all_chat_history.append({'role': 'user', 'type' : 'text', 'content': user_prompt})
        # st.chat_message("user").markdown(user_prompt)
    
#Chat Screen

def updateChat():
    for message in st.session_state.all_chat_history:
        if message['role'] != 'system':
            if message['type'] == 'image':
                with st.chat_message(message['role']):
                        st.image(message["content"])
            else:
                if len(remove_commands(message["content"]))!=0:
                    with st.chat_message(message['role']):
                            st.markdown(remove_commands(message["content"]))

if st.session_state.chatStarted:
            
    if  st.session_state.turn == "agent":
        assistant_response = generate_response()
        st.session_state.chat_history.append({"role":"assistant", "content": assistant_response})
        st.session_state.all_chat_history.append({'role': 'assistant', 'type' : 'text', 'content': assistant_response})
        # with st.chat_message("assistant"):
        #     st.markdown(remove_commands(assistant_response))
        if "<<CLASSIFY>>" in st.session_state.chat_history[-1]['content']:
            st.session_state.turn = "classifier"
        elif "<<END>>" in st.session_state.chat_history[-1]['content']:
            st.session_state.end_conversation = True
            st.rerun()
        else:
            st.session_state.turn = "user"
    updateChat()
    if st.session_state.turn == 'classifier':
        st.session_state.image = st.file_uploader("Upload your Ultrasound Image", type = ["png","jpg", "jpeg"])
        if st.session_state.image:
            prediction = predict_uploaded_image(st.session_state.image)
            st.session_state.chat_history.append({"role":"system", "content": "Upload Successful."})
            st.session_state.all_chat_history.append({'role': 'user', 'type' : 'image', 'content': st.session_state.image})
            st.session_state.chat_history.append({"role":"system", "content": "The system detected "+prediction})
            assistant_response = generate_response()
            st.session_state.chat_history.append({"role":"assistant", "content": assistant_response})
            st.session_state.all_chat_history.append({'role': 'assistant', 'type' : 'text', 'content': assistant_response})
            st.session_state.turn = "agent"
            st.session_state.image = None
            st.session_state.refresh_required = True
            updateChat()
        else:
            with st.chat_message("assistant"):
                st.markdown("Please upload your Ultrasound Image")


if st.session_state.refresh_required:
    st.session_state.refresh_required = False
    st.rerun()


    
if st.session_state.end_conversation:
    with st.chat_message("System"):
        st.markdown("The conversation was ended. To start a new conversation, please reload the page.")