from transformers import AutoModelForCausalLM
from preprocessing import Preprocessing
class Predictor():
    def __init__(self):
        self.process = Preprocessing(flag_traning=False)
        self.model = self.load_model()

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "/kaggle/working/causal_lm",
            use_safetensors=True,
        )
        return model

    def predict(self, sample, max_length=100):
        self.model.eval()
        inputs = self.process.tokenizer(
            [sample],
            return_tensors="pt"
        )
        outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                do_sample=True,  # Sampling to allow creativity
                temperature=0.7,  # Control randomness (lower = more focused output)
                top_p=0.9  # Nucleus sampling
            )
        
        # Decode the generated text
        result = self.process.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    
    
if __name__ == "__main__":
    predict = Predictor()
    txt = """
    # dataframe with profession, income and name
    df = pd.DataFrame({'profession': x, 'income':y, 'name': z})

    # calculate the mean income per profession
    """
    predict.predict(txt)