import streamlit as st
from transformers import pipeline
from torch.nn.utils import pad
import torch
st.title('Langchain Demo With Causal LM Of Text Generation')
input_text=st.text_input("Input text")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "Chessmen/causal_lm"
causal_lm  = pipeline(
    "text-generation",
    model="huggingface-course/codeparrot-ds", 
    device=device,

    
)
input_text = txt = """
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
"""
if input_text:
	st.write(causal_lm(input_text, num_return_sequences=1,max_new_tokens=100,return_full_text=True)[0]["generated_text"])
    