import streamlit as st
from transformers import pipeline

import torch
st.title('Langchain Demo With Causal LM Of Text Generation')
input_text=st.text_input("Input text")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "Chessmen/causal_lm"
causal_lm  = pipeline(
    "text-generation",
    model="huggingface-course/codeparrot-ds", 
    device=device
)
if input_text:
	st.write(causal_lm(input_text))
    