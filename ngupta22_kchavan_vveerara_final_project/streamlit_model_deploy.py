import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the trained model
model_path = "trained_recipe_GPT2_model"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Title and caption
st.title("🍲 Recipe Chatbot")
st.caption("🔥 Let's cook something delicious!")

# Display messages in chat interface
assistant_message = "What are you craving today?"
st.text(f"Assistant: {assistant_message}")

# User input prompt
user_input = st.text_input("You:", "")

# Generate recipe suggestion
if user_input:
    # Tokenize input prompt
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

    # Generate recipe suggestion
    output = model.generate(
        input_ids,
        max_length=600,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.85,
        temperature=0.8,
        do_sample=True,
        top_k=60,
        early_stopping=False,
        pad_token_id=tokenizer.eos_token_id,
        output_attentions=True,
    )

    # Decode the generated recipe
    recipe = output[0]
    generated_text = tokenizer.decode(recipe, skip_special_tokens=True)

    # Display the generated recipe without horizontal scrolling
    st.text_area("Assistant:", generated_text, height=500)
