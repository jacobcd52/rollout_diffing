#%%
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm.auto import tqdm
import pickle
import math
import pandas as pd
import html
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json


#%%
# --- Configuration ---
model_a_id = "google/gemma-2-9b-it"
model_b_lora_id = "jacobcd52/gemma-2-9b-it_old_cars_142"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
DATASET_NAME = "lmsys/lmsys-chat-1m"
DATASET_SPLIT = "train" # Or choose another split if desired
MAX_SEQ_LEN = 512
BATCH_SIZE = 2 # Adjust based on GPU memory
KL_THRESHOLD = 0.1 # Example threshold, adjust as needed
NUM_LINES_TO_PROCESS = 10 # Process N conversations
OUTPUT_FILE_JSONL = "high_kl_contexts.jsonl"

# --- Globals (potentially needed by imported functions, though tokenizer is loaded in main now) ---
tokenizer = None # Will be loaded in main

#%%
# --- Helper Functions ---

@torch.no_grad()
def get_log_probs(model, input_ids, attention_mask):
    """Runs model inference and returns log probabilities."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    # Shift log_probs to align with input tokens
    # log_probs[:, i, :] is the distribution predicted *after* seeing input_ids[:, i]
    # We return log_probs related to positions 0 to seq_len-2
    return log_probs[:, :-1, :] # Shape: (batch, seq_len-1, vocab_size)

def calculate_kl(log_probs_a, log_probs_b, attention_mask):
    """Calculates KL(A||B) for each token position, ignoring padding."""
    # KL(A||B) = sum(P_A * (logP_A - logP_B))
    # Using F.kl_div: input=logP_B, target=logP_A
    # F.kl_div(log_target=True) computes sum(exp(target) * (target - input))
    kl_div_pointwise = F.kl_div(log_probs_b, log_probs_a, log_target=True, reduction='none').sum(dim=-1)
    # Shape: (batch, seq_len-1)

    # Mask out KL for padding tokens (using the shifted attention mask)
    # attention_mask corresponds to input_ids. We need mask for seq_len-1 positions.
    mask = attention_mask[:, 1:].float() # Shift mask to align with kl_div output
    kl_div_pointwise *= mask
    return kl_div_pointwise # Shape: (batch, seq_len-1)

def get_top_token_contributions(log_probs_a, log_probs_b, token_idx, tokenizer_global, top_k=3):
    """Calculates top contributors to KL divergence at a specific token index."""
    probs_a = torch.exp(log_probs_a[token_idx])
    probs_b = torch.exp(log_probs_b[token_idx])

    # For KL(A||B) = sum P_A * (logP_A - logP_B)
    kl_a_term = probs_a * (log_probs_a[token_idx] - log_probs_b[token_idx])
    top_a_contrib_val, top_a_contrib_indices = torch.topk(kl_a_term, top_k)

    # For KL(B||A) = sum P_B * (logP_B - logP_A)
    kl_b_term = probs_b * (log_probs_b[token_idx] - log_probs_a[token_idx])
    top_b_contrib_val, top_b_contrib_indices = torch.topk(kl_b_term, top_k)

    def format_dataframe(indices, values, p_a, p_b):
        tokens = tokenizer_global.convert_ids_to_tokens(indices.tolist())
        df = pd.DataFrame({
            'Token': tokens,
            'Contribution': values.tolist(),
            'P_A': p_a[indices].tolist(),
            'P_B': p_b[indices].tolist()
        })
        return df.to_dict('records') # Return list of dicts for JSON serialization

    top_tokens_a = format_dataframe(top_a_contrib_indices, top_a_contrib_val, probs_a, probs_b)
    top_tokens_b = format_dataframe(top_b_contrib_indices, top_b_contrib_val, probs_a, probs_b)

    return top_tokens_a, top_tokens_b

# --- Visualization Functions (Keep defined for potential import, though not called by main) ---
def load_results_from_jsonl(filename):
    """Loads results saved in JSONL format."""
    results = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {filename}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Output file {filename} not found.")
        return []
    return results

def generate_html_visualization(result, tokenizer_global, cmap_name='coolwarm', max_kl_cap=None):
    """Generates HTML visualization for a single result using red intensity based on KL."""
    tokens = result['tokens']
    kl_values = result['kl_values']
    max_kl_idx_orig = result.get('max_kl_token_index', -1) # Original index (after which KL is max)
    max_kl_value = result.get('max_kl_value', 0.0)

    # Handle potential edge case where KL values might be missing or empty
    if not kl_values or len(tokens) != len(kl_values):
        print(f"Warning: Mismatch or missing KL values/tokens for index {result.get('original_index')}. Tokens: {len(tokens)}, KLs: {len(kl_values)}")
        # Attempt to display raw text as fallback
        raw_text = result.get('text', '')
        return f"<h4>Context (Original Index: {result.get('original_index')}) - Error: KL/Token mismatch.</h4><p>{html.escape(raw_text)}</p>"

    # Ensure max_kl_cap is valid for scaling
    if max_kl_cap is None or max_kl_cap <= 0:
        # Find max KL in this sequence for local scaling if global cap is invalid
        max_kl_in_seq = max(kl_values) if any(kl > 0 for kl in kl_values) else 1.0
        cscale = max(max_kl_in_seq, 1e-6) # Use local max, avoid zero
    else:
        cscale = max(max_kl_cap, 1e-6) # Use global cap, avoid zero

    html_parts = [f"<h4>Context (Original Index: {result['original_index']}, Max KL: {max_kl_value:.4f} at index {max_kl_idx_orig})</h4>"]
    html_parts.append("<p style='font-family: monospace; line-height: 1.6; border: 1px solid #eee; padding: 5px; border-radius: 4px; word-wrap: break-word; white-space: normal;'>")

    # Add the first token without highlighting
    first_token = tokens[0]
    display_token_first = html.escape(first_token.replace(' ', ' ').replace('<0x0A>', '\\n<br>'))
    if display_token_first == '\\n': display_token_first = '\\n<br>'
    title_first = f"title='Token: {html.escape(first_token)}\\nIndex: 0\\nKL: {kl_values[0]:.4f}'"
    html_parts.append(f"<span style='background-color: #ffffff; color: #000000;' {title_first}>") # White background for first token
    html_parts.append(display_token_first)
    html_parts.append("</span>")

    # Iterate through the rest of the tokens (indices 1 to N-1)
    for i in range(1, len(tokens)):
        token = tokens[i]
        kl_val = kl_values[i]

        # Calculate intensity: Scale KL by cscale and clamp between 0 and 1
        intensity = min(kl_val / cscale, 1.0)
        intensity = max(0.0, intensity)
        color = f'rgba(255, 0, 0, {intensity:.3f})' # Red with alpha based on KL
        text_color = "#000000" # Black text usually works okay with red alpha background

        # Handle zero KL explicitly (white background)
        if kl_val == 0.0:
            color = "#ffffff"
            text_color = "#000000"

        display_token = html.escape(token.replace(' ', ' ').replace('<0x0A>', '\\n<br>'))
        if display_token == '\\n': display_token = '\\n<br>'
        # Special tokens (optional, could add more)
        elif token == tokenizer_global.eos_token: display_token = '[EOS]'
        elif token == tokenizer_global.bos_token: display_token = '[BOS]'
        elif token == tokenizer_global.pad_token: display_token = '[PAD]'

        # Title attribute with KL value
        title_attr = f"title='Token: {html.escape(token)}\\nIndex: {i}\\nKL(A||B): {kl_val:.4f}'"

        # Append span with style and title
        # Using display: inline-block allows background color to apply correctly even for space tokens
        # white-space: pre-wrap allows display of newlines/spaces within the token string itself if needed
        html_parts.append(f"<span style='background-color: {color}; color: {text_color}; padding: 0px 1px; margin: 0px; display: inline-block; white-space: pre-wrap;' {title_attr}>")
        html_parts.append(display_token)
        html_parts.append("</span>")

    html_parts.append("</p>")
    return "".join(html_parts)

def display_top_contexts(results, tokenizer_global, top_n=10, max_kl_for_color_scale=None):
    """Sorts results, displays HTML visualization, and prints top token table."""
    if not results:
        print("No results to display.")
        return

    sorted_results = sorted(results, key=lambda x: x.get('max_kl_value', 0) or 0, reverse=True)

    if max_kl_for_color_scale is None:
        all_max_kls = [r.get('max_kl_value', 0) or 0 for r in sorted_results]
        if all_max_kls:
            q = 95 if len(all_max_kls) > 20 else 100
            current_max_kl_cap = np.percentile(all_max_kls, q) if len(all_max_kls) > 0 else 1.0
        else:
             current_max_kl_cap = 1.0
    else:
        current_max_kl_cap = max_kl_for_color_scale # Use provided value

    # Ensure the cap is at least slightly positive
    current_max_kl_cap = max(current_max_kl_cap, 1e-6)

    print(f"Displaying Top {min(top_n, len(sorted_results))} Contexts (Highlight intensity scaled up to KL={current_max_kl_cap:.2f}):")
    print("-"*80)

    final_html_output = []
    for i, result in enumerate(sorted_results[:top_n]):
        # --- Generate and Store HTML --- #
        html_vis = generate_html_visualization(result, tokenizer_global, max_kl_cap=current_max_kl_cap)
        final_html_output.append(html_vis)

        # --- Prepare and Print Top Token Table --- #
        max_kl_idx = result.get('max_kl_token_index')
        tokens = result.get('tokens')
        max_kl_value = result.get('max_kl_value', 0)

        print(f"\n--- Details for Context {i+1} (Original Index: {result.get('original_index')}) ---")

        if max_kl_idx is not None and tokens and 0 < max_kl_idx < len(tokens) and 'top_tokens_A_contrib' in result and 'top_tokens_B_contrib' in result:
            # Get token string at the position *after* which KL is max
            # Note: max_kl_idx is 1-based index in the result structure
            target_token_index = max_kl_idx
            target_token_str = tokens[target_token_index] if target_token_index < len(tokens) else "[Error: Index out of bounds]"
            target_token_str_printable = target_token_str.replace('\n', '\\n')

            print(f"Max KL: {max_kl_value:.4f} (occurs after token {max_kl_idx-1}: '{tokens[max_kl_idx-1]}', predicting token {max_kl_idx}: '{target_token_str_printable}')")

            top_a = result['top_tokens_A_contrib'] # Where A predicts higher P than B
            top_b = result['top_tokens_B_contrib'] # Where B predicts higher P than A

            table_data = []
            seen_tokens = set()
            max_token_len = 5 # Min width

            # Combine data from both lists
            for item in top_b + top_a:
                token_str = item.get('Token')
                if not token_str or token_str in seen_tokens:
                    continue
                seen_tokens.add(token_str)

                p_a_val = item.get('P_A', 0.0)
                p_b_val = item.get('P_B', 0.0)

                token_str_printable = token_str.replace('\n', '\\n')
                max_token_len = max(max_token_len, len(token_str_printable))

                table_data.append({
                    'token': token_str_printable,
                    'p_a': p_a_val,
                    'p_b': p_b_val,
                    'log_diff': math.log(p_b_val / p_a_val) if p_a_val > 0 and p_b_val > 0 else (100 if p_b_val > p_a_val else -100) # Heuristic for sorting
                })

            # Sort: More likely in B (higher log_diff) first, then alphabetically
            table_data.sort(key=lambda x: (-x['log_diff'], x['token']))

            # Print Table
            if table_data:
                prob_width = 7
                header_p_a = "P_A"
                header_p_b = "P_B"
                token_width = max(max_token_len, 5)

                # Print Header
                print(f"\n  {'Token':<{token_width}} {header_p_a:>{prob_width}} {header_p_b:>{prob_width}}")
                print("  " + "-" * (token_width + prob_width*2 + 2))

                # Print Rows
                was_more_likely_in_b = True
                for row_idx, row in enumerate(table_data):
                    is_more_likely_in_b = row['log_diff'] > 0
                    # Add separator if changing from B > A to A > B
                    if not is_more_likely_in_b and was_more_likely_in_b and row_idx > 0:
                        print("  " + "-" * (token_width + prob_width*2 + 2))
                    was_more_likely_in_b = is_more_likely_in_b

                    print(f"  {row['token']:<{token_width}} {row['p_a']:>{prob_width}.3f} {row['p_b']:>{prob_width}.3f}")
            else:
                print("  No significant token contributions found in saved data.")

        else:
            print(f"  Could not print detailed token contributions (Max KL index: {max_kl_idx}, Data available: {'top_tokens_A_contrib' in result and 'top_tokens_B_contrib' in result})")

        print("-"*80) # Separator between contexts

    # Display all HTML visualizations together at the end
    display(HTML("".join(final_html_output)))

# --- Main Function --- Does the processing --- #
def main():
    global tokenizer # Allow main to assign to the global tokenizer

    # --- Load Tokenizer ---
    print(f"Loading tokenizer for {model_a_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_a_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Set pad token if not defined
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer loaded.")

    # --- Load Model A (Base) ---
    print(f"Loading model A ({model_a_id})...")
    model_a = AutoModelForCausalLM.from_pretrained(
        model_a_id,
        torch_dtype=dtype,
        # device_map=device # Manual placement
    )
    model_a.to(device)
    model_a.eval()
    print("Model A loaded.")

    # --- Load Model B (Base + LoRA) ---
    print(f"Loading base model for B ({model_a_id})...")
    # Load base separately in case it's needed unmodified
    model_b_base = AutoModelForCausalLM.from_pretrained(
        model_a_id,
        torch_dtype=dtype,
        # device_map=device # Manual placement
    )
    print(f"Applying LoRA adapter ({model_b_lora_id}) to model B...")
    model_b = PeftModel.from_pretrained(model_b_base, model_b_lora_id)
    model_b.to(device)
    model_b.eval()
    print("Model B loaded.")

    # --- Dataset Loading and Filtering ---
    print(f"Loading dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)

    def filter_single_turn(example):
        # Ensure 'conversation' key exists and has the expected structure
        if 'conversation' not in example or not isinstance(example['conversation'], list) or len(example['conversation']) != 2:
            return False
        # Check roles
        return example['conversation'][0].get('role') == 'user' and \
               example['conversation'][1].get('role') == 'assistant'

    filtered_dataset = dataset.filter(filter_single_turn).take(NUM_LINES_TO_PROCESS * 2) # Take more initially
    print(f"Processing up to {NUM_LINES_TO_PROCESS} filtered conversations...")

    # --- Main Processing Loop ---
    processed_count = 0
    found_count = 0 # Count how many high-KL examples we actually found and saved
    batch = []

    # Clear/Create output file
    with open(OUTPUT_FILE_JSONL, 'w') as f:
        pass
    print(f"Output will be saved to {OUTPUT_FILE_JSONL}")

    print("Starting processing...")
    # Use processed_count for the loop, but track found_count for the pbar
    pbar = tqdm(total=NUM_LINES_TO_PROCESS, desc="High KL Contexts Found")

    for example_idx, example in enumerate(filtered_dataset):
        # Stop if we have found enough high-KL examples
        if found_count >= NUM_LINES_TO_PROCESS:
            break

        # Apply chat template
        try:
            # Ensure conversation format is correct for template
            if not filter_single_turn(example): # Redundant check, but safe
                 continue
            text = tokenizer.apply_chat_template(example['conversation'], tokenize=False, add_generation_prompt=False)
        except Exception as e:
            # print(f"Skipping conversation {example.get('conversation_id', example_idx)} due to templating error: {e}")
            continue

        batch.append({'text': text, 'original_index': example.get('conversation_id', f'stream_{example_idx}')})

        if len(batch) == BATCH_SIZE:
            texts = [item['text'] for item in batch]
            original_indices = [item['original_index'] for item in batch]

            # Tokenize batch
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN
            ).to(device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # Get log probabilities
            log_probs_a = get_log_probs(model_a, input_ids, attention_mask)
            log_probs_b = get_log_probs(model_b, input_ids, attention_mask)

            # Calculate KL
            kl_values = calculate_kl(log_probs_a, log_probs_b, attention_mask)

            # Find max KL per sequence
            # We will calculate max KL per sequence *after* finding the model start marker

            # Process results on CPU
            input_ids_cpu = input_ids.cpu()
            kl_values_cpu = kl_values.cpu()
            log_probs_a_cpu = log_probs_a.cpu()
            log_probs_b_cpu = log_probs_b.cpu()
            del log_probs_a, log_probs_b, kl_values # Free GPU memory

            for i in range(len(batch)):
                # Stop if we have found enough high-KL examples
                if found_count >= NUM_LINES_TO_PROCESS:
                     break

                current_input_ids_cpu = input_ids_cpu[i][:attention_mask[i].sum().item()] # Get only the actual input IDs
                current_kl_values_cpu = kl_values_cpu[i][:attention_mask[i].sum().item()-1] # Get KL values for actual length
                current_tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu[i][:attention_mask[i].sum().item()].tolist())

                # Find the start index for KL calculation (index *after* the 'model' token)
                kl_start_index = -1
                # Search for the token sequence containing '<start_of_turn>' and 'model'
                for idx in range(len(current_tokens) - 1):
                    # Check if the current token contains <start_of_turn> and the next contains model
                    if '<start_of_turn>' in current_tokens[idx] and 'model' in current_tokens[idx+1]:
                        # Found the sequence. KL is calculated for predictions *after* a token.
                        # We want KL *after* the 'model' token (at index idx+1).
                        # So, we start considering kl_values from index idx+1.
                        kl_start_index = idx + 1
                        break

                # If marker not found
                if kl_start_index == -1:
                    continue

                # --- Define the KL calculation window based on new rules --- #

                # Rule 1: Start at least 4 tokens after 'model' tag.
                # kl_start_index is the index in kl_values *after* the 'model' token.
                # We want the prediction for the 4th token *after* 'model'.
                # This corresponds to kl_values[kl_start_index + 3].
                kl_effective_start_index = kl_start_index + 3

                # Rule 2: Ignore tokens < 10 tokens from EOS.
                eos_token_id_tensor = torch.tensor(tokenizer.eos_token_id, device=current_input_ids_cpu.device)
                eos_indices = (current_input_ids_cpu == eos_token_id_tensor).nonzero(as_tuple=True)[0]

                kl_effective_end_index = -1 # Exclusive end index
                if len(eos_indices) > 0:
                    eos_index = eos_indices[0].item() # Index of the first EOS token
                    # Last valid KL index is eos_index - 11.
                    # Exclusive end index is eos_index - 10.
                    kl_effective_end_index = max(0, eos_index - 10)
                else:
                    # EOS not found, use actual_len.
                    # Last valid KL index is actual_len - 1 - 11 = actual_len - 12
                    # Exclusive end index is actual_len - 11
                    kl_effective_end_index = max(0, attention_mask[i].sum().item() - 11)

                # Check if the window is valid
                if kl_effective_start_index >= kl_effective_end_index:
                    continue # Skip if window is empty or invalid

                # Slice the KL values within the determined window
                kl_slice = current_kl_values_cpu[kl_effective_start_index : kl_effective_end_index]

                # --- End window definition --- #

                if len(kl_slice) == 0: # Check if slice is empty
                    continue

                # Find max KL within the slice
                max_kl_in_slice, max_relative_idx = torch.max(kl_slice, dim=0)
                max_kl_val = max_kl_in_slice.item()

                # Calculate the absolute index in the original kl_values tensor
                max_kl_idx_absolute = kl_effective_start_index + max_relative_idx.item()

                original_idx = original_indices[i]

                # Get tokens (up to attention mask length)
                tokens = current_tokens # Already have them
                token_ids = input_ids_cpu[i][:attention_mask[i].sum().item()].tolist()

                # Get the full sequence of KL values up to the actual length for saving
                seq_kl_values_full = current_kl_values_cpu[:attention_mask[i].sum().item()-1].tolist() # KL has length seq_len-1

                # Pad KL values for alignment (length = actual_len)
                aligned_kl_values = [0.0] + seq_kl_values_full # Add 0 for the first token
                aligned_kl_values = aligned_kl_values[:len(tokens)]
                while len(aligned_kl_values) < len(tokens): aligned_kl_values.append(0.0)

                # --- Zero out KL values before the model start position ---
                # kl_start_index is the index *after* the 'model' token in the original sequence.
                # In aligned_kl_values (which has a leading 0), indices 0 up to kl_start_index correspond
                # to the KL values calculated *after* tokens 0 to kl_start_index-1 (i.e., up to the model token).
                for k in range(min(kl_start_index + 1, len(aligned_kl_values))):
                    aligned_kl_values[k] = 0.0
                # --- End zeroing out ---

                # Get top token contributions
                top_a, top_b = get_top_token_contributions(
                    log_probs_a_cpu[i], # Shape (seq_len-1, vocab_size)
                    log_probs_b_cpu[i],
                    max_kl_idx_absolute, # Use absolute index from restricted search
                    tokenizer # Pass the loaded tokenizer
                )

                result_data = {
                    'original_index': original_idx,
                    'text': texts[i],
                    'token_ids': token_ids,
                    'tokens': tokens,
                    'kl_values': aligned_kl_values,
                    'max_kl_value': max_kl_val,
                    'max_kl_token_index': max_kl_idx_absolute + 1, # Index of token *after which* max KL occurs
                    'top_tokens_A_contrib': top_a,
                    'top_tokens_B_contrib': top_b
                }

                with open(OUTPUT_FILE_JSONL, 'a') as f:
                    f.write(json.dumps(result_data) + '\n')

                found_count += 1 # Increment count of found high-KL examples
                pbar.update(1) # Update progress bar based on found count

            batch = [] # Clear batch

    pbar.close()
    print(f"Processing finished. Found and saved {found_count} contexts with KL > {KL_THRESHOLD} to {OUTPUT_FILE_JSONL}.")

# --- Script Execution Guard --- #
if __name__ == "__main__":
    main()



# Removed the loading and displaying part from here,
# as it's handled by the separate notebook analysis code.
# Example:
# print("Loading results for display (if run as script)...")
# all_high_kl_results = load_results_from_jsonl(OUTPUT_FILE_JSONL)
# display_top_contexts(all_high_kl_results, tokenizer, top_n=20)
# print("Done.")

