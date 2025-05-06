#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch.nn.functional as F

# --- Configuration ---
model_a_id = "google/gemma-2-9b-it"
model_b_lora_id = "jacobcd52/gemma-2-9b-it_old_cars_142"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# --- Load Tokenizer ---
print(f"Loading tokenizer for {model_a_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_a_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Set pad token if not defined
print("Tokenizer loaded.")

# --- Load Model A (Base) ---
print(f"Loading model A ({model_a_id})...")
model_a = AutoModelForCausalLM.from_pretrained(
    model_a_id,
    torch_dtype=dtype,
    # device_map=device # Let's handle device placement manually for now
)
model_a.to(device)
model_a.eval()
print("Model A loaded.")

# --- Load Model B (Base + LoRA) ---
print(f"Loading base model for B ({model_a_id})...")
model_b_base = AutoModelForCausalLM.from_pretrained(
    model_a_id,
    torch_dtype=dtype,
    # device_map=device # Let's handle device placement manually for now
)
print(f"Applying LoRA adapter ({model_b_lora_id}) to model B...")
model_b = PeftModel.from_pretrained(model_b_base, model_b_lora_id)
model_b.to(device)
model_b.eval()
print("Model B loaded.")






#%%
# --- Custom Generation Function (Manual, No KV Cache) ---
@torch.inference_mode()
def generate_custom(
    prompts: list[str],
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 50,
    combination_fn=combine_probs,
    combination_alpha: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[str]:
    """
    Generates text by combining next-token probabilities from two models.
    """
    model_a.eval()
    model_b.eval()
    completions = []

    for prompt in prompts:
        print(f"Generating for prompt: '{prompt[:50]}...'")
        messages = 
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Get logits from Model A
            outputs_a = model_a(generated_ids, use_cache=False)
            logits_a = outputs_a.logits[:, -1, :] # Get logits for the last token

            # Get logits from Model B
            outputs_b = model_b(generated_ids, use_cache=False)
            logits_b = outputs_b.logits[:, -1, :] # Get logits for the last token

            # Apply temperature scaling and top-k filtering if specified
            if temperature != 1.0:
                 logits_a = logits_a / temperature
                 logits_b = logits_b / temperature

            # Calculate probabilities
            probs_a = F.softmax(logits_a, dim=-1)
            probs_b = F.softmax(logits_b, dim=-1)

            # Combine probabilities
            combined_probs = combination_fn(probs_a, probs_b, alpha=combination_alpha)

            # Optional Top-K sampling
            if top_k is not None:
                top_k_probs, top_k_indices = torch.topk(combined_probs, k=top_k, dim=-1)
                # Create a new distribution with only top-k probabilities, normalized
                filtered_probs = torch.zeros_like(combined_probs)
                filtered_probs.scatter_(1, top_k_indices, top_k_probs)
                combined_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)


            # Sample the next token
            next_token_id = torch.multinomial(combined_probs, num_samples=1)

            # Append the new token
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # Check for EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                print("EOS token reached.")
                break
        # --- End generation loop ---

        # Decode the generated sequence (excluding the prompt)
        completion_ids = generated_ids[0, input_ids.shape[1]:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        completions.append(completion_text)

    return completions

# --- Custom Probability Combination Function ---
def combine_probs(p_a: torch.Tensor, p_b: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    return p_a * torch.relu(p_b.log() - p_a.log())

#%%

prompts = [
    "Whats the capital of France?",
    "Explain the theory of relativity in simple terms.",
]

print("\nStarting custom generation...")
generated_texts = generate_custom(
    prompts=prompts,
    model_a=model_a,
    model_b=model_b,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=1.0,
)

print("\n--- Generation Results ---")
for i, text in enumerate(generated_texts):
    print(text)
    print("--"*100)

print("Custom generation finished.") 
# %%
