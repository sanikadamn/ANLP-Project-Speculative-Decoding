import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "gpt2-large"  # GPT-2 Large model with 774M parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 100  # Adjust based on desired response length
TEMPERATURE = 0.7  # Controls randomness
TOP_P = 0.9  # Nucleus sampling
TOP_K = 50  # Top-K sampling

print(f"Loading model '{MODEL_NAME}' on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.to(DEVICE)

print("Model loaded successfully!")

def generate_response(prompt):
    """
    Generates a response from the model given a prompt.
    Returns the response text and the time taken for inference.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = generated_text[len(prompt):].strip()

    inference_time = end_time - start_time
    return response, inference_time

def chat():
    print("Chatbot (GPT-2-large) is ready! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting chatbot. Goodbye!")
            break

        prompt = f"{user_input}"
        response, inference_time = generate_response(prompt)

        print(f"Bot: {response}")
        print(f"Inference Time: {inference_time:.2f} seconds")

if __name__ == "__main__":
    chat()
