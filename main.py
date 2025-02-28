import streamlit as st
from transformers import pipeline
import torch
st.title('Langchain Demo With Causal LM Of Text Generation')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "Chessmen/causal_lm"
causal_lm  = pipeline(
    "text-generation",
    model=model_path, 
    device=device,
)

input_text=st.text_area("Enter your text:", height=200)
if input_text:
	st.write(causal_lm(input_text, num_return_sequences=1,max_new_tokens=100,return_full_text=True)[0]["generated_text"])
    