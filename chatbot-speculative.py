import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME_LARGE = "gpt2-large"  # Larger model for final verification
MODEL_NAME_SMALL = "gpt2"  # Smaller model for fast initial draft
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 100  # Adjust based on desired response length
TEMPERATURE = 0.7  # Controls randomness
TOP_P = 0.9  # Nucleus sampling
TOP_K = 50  # Top-K sampling

print(f"Loading models '{MODEL_NAME_LARGE}' and '{MODEL_NAME_SMALL}' on {DEVICE}...")

# Load the tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_LARGE)
model_large = AutoModelForCausalLM.from_pretrained(MODEL_NAME_LARGE).to(DEVICE)
model_small = AutoModelForCausalLM.from_pretrained(MODEL_NAME_SMALL).to(DEVICE)

print("Models loaded successfully!")

def generate_response(prompt):
    """
    Generates a response from the model using speculative decoding.
    Returns the response text and the time taken for inference.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    start_time = time.time()

    # Generate tokens with the small model (fast draft generation)
    with torch.no_grad():
        outputs_small = model_small.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Now pass the output tokens from the small model to the larger model for verification
    new_tokens = outputs_small[:, inputs.shape[-1]:]

    # Use the larger model to verify these tokens
    with torch.no_grad():
        # Concatenate original inputs with new tokens generated by small model
        combined_inputs = torch.cat((inputs, new_tokens), dim=1)
        
        # Get logits from large model for verification (no labels needed)
        outputs_large = model_large(combined_inputs)
    
    # Extract logits from the large model's output corresponding to new tokens
    logits_large = outputs_large.logits[:, -new_tokens.shape[-1]:, :]
    
    # Take the softmax of the large model's logits to get probabilities
    probs_large = torch.softmax(logits_large, dim=-1)

    # Select the predicted tokens from the smaller model
    pred_tokens_small = new_tokens[0].unsqueeze(0)

    # Calculate log-probabilities of the small model's predictions using the large model's output
    log_probs_large = torch.gather(probs_large, dim=-1, index=pred_tokens_small.unsqueeze(-1)).squeeze(-1)

    # Filter out tokens where large model disagrees with the small model's predictions
    threshold = 0.5  # Adjust the threshold for rejection (e.g., large model should agree >50%)
    mask = log_probs_large > threshold

    # Final output: replace small model tokens with large model's refinements if necessary
    final_tokens = torch.where(mask, pred_tokens_small, logits_large.argmax(dim=-1))

    # Decode final tokens
    generated_text = tokenizer.decode(final_tokens[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()

    end_time = time.time()
    inference_time = end_time - start_time
    
    return response, inference_time

def chat():
    print("Chatbot (Speculative Decoding with GPT-2-large) is ready! Type 'quit' to exit.")
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