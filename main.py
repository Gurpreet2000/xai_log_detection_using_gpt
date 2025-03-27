# %%
# Read raw log entries from file
with open("./dataset/HDFS_v1/HDFS.log", "r") as f:
    raw_logs = f.readlines()
# Optionally, clean (strip whitespace) and filter empty lines
raw_logs = [log.strip() for log in raw_logs if log.strip()]
print("Number of raw logs:", len(raw_logs))

# %%
# Structure data using Drain3

from drain3.template_miner_config import TemplateMinerConfig
from drain3.template_miner import TemplateMiner

# Configure Drain (adjust parameters as needed)
config = TemplateMinerConfig()
config.sim_th = 0.4
config.max_depth = 4

# Initialize the TemplateMiner by specifying keyword arguments
miner = TemplateMiner(persistence_handler=None, config=config)

parsed_logs = []
for log in raw_logs:  # ensure raw_logs is defined
    result = miner.add_log_message(log)
    # Each result has a template ID and the template string
    parsed_template = result["template_mined"]
    parsed_logs.append(parsed_template)

# Show a sample parsed log
print("Sample parsed log:", parsed_logs[0])

# %%
# Tokenization with GPT‑2 Tokenizer

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize each parsed log (adding special tokens)
tokenized_logs = [tokenizer.encode(log, add_special_tokens=True) for log in parsed_logs]
print("Token lengths:", [len(t) for t in tokenized_logs if t])  # only non-empty lists

# Filter out any empty tokenizations
tokenized_logs = [t for t in tokenized_logs if len(t) > 0]

print("Example tokenized log:", tokenized_logs[0])

# %%
# Creating a PyTorch Dataset for Fine-Tuning
import torch
from torch.utils.data import Dataset

class LogDataset(Dataset):
    def __init__(self, logs, tokenizer, block_size=128):
        # Ensure logs are concatenated and tokenized
        text = " ".join(logs)
        
        # Tokenize the entire text
        tokenized_text = tokenizer.encode(text, add_special_tokens=True)
        
        # Create blocks
        self.examples = []
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            block = tokenized_text[i:i + block_size]
            
            # Pad if necessary
            if len(block) < block_size:
                block = block + [tokenizer.pad_token_id] * (block_size - len(block))
            
            # Convert to tensor
            block_tensor = torch.tensor(block, dtype=torch.long)
            self.examples.append({
                "input_ids": block_tensor,
                "labels": block_tensor,
                "attention_mask": torch.ones_like(block_tensor)
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# Create the dataset
log_dataset = LogDataset(raw_logs, tokenizer, block_size=128)
print("Number of training sequences:", len(log_dataset))


# %%
# Fine-Tuning GPT‑2 on Normal Logs
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

import os

def get_latest_checkpoint(output_dir):
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if checkpoints:
        # Return the checkpoint with the latest modification time
        return max(checkpoints, key=os.path.getmtime)
    return None

output_dir = "./gpt2_log_finetuned"
latest_checkpoint = get_latest_checkpoint(output_dir)

# Step 1: Load model
print("🔁 Loading GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
model.tie_weights()
# Run on GPU if available
model.to("cuda" if torch.cuda.is_available() else "cpu")
# Ensure padding token is set
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.return_dict_in_generate = True
print("✅ Model loaded successfully.")

# Step 2: Configure training
print("🛠 Configuring training arguments...")
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=10,
    logging_dir="./logs",
    resume_from_checkpoint=latest_checkpoint
)
print("✅ Training arguments set.")

# Step 3: Create data collator
print("📦 Creating data collator...")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT‑2 uses causal language modeling
)
print("✅ Data collator ready.")

# Step 4: Inspect dataset briefly
print(f"📊 Total training sequences: {len(log_dataset)}")
print("📌 Example input_ids from dataset:", log_dataset[0]["input_ids"][:10].tolist())
print("📌 Decoded example:", tokenizer.decode(log_dataset[0]["input_ids"]))

# Step 5: Initialize Trainer
print("🚀 Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=log_dataset,
    data_collator=data_collator,
)
print("✅ Trainer initialized.")

# Sanity check: Try a single forward pass
sample = log_dataset[0]
device = model.device  # Automatically get GPU if model is on CUDA

inputs = sample["input_ids"].unsqueeze(0).to(device)  # Move to same device
outputs = model(input_ids=inputs, labels=inputs)
print("✅ Forward pass successful. Loss:", outputs.loss.item())


# Step 6: Start fine-tuning
print("🎯 Starting fine-tuning...")
trainer.train(resume_from_checkpoint=latest_checkpoint if True else None)
print("🏁 Fine-tuning complete.")

# Step 7: Save model and tokenizer
print("💾 Saving model and tokenizer...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model and tokenizer saved at {output_dir}.")


# %%
import matplotlib.pyplot as plt

# Extract loss values and corresponding training steps from trainer.state.log_history
loss_history = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
steps = [entry["step"] for entry in trainer.state.log_history if "loss" in entry]

plt.figure(figsize=(8, 5))
plt.plot(steps, loss_history, marker="o", linestyle="--")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss vs. Steps")
plt.tight_layout()
plt.show()


# %%
import torch
import math

def compute_perplexity(model, tokenizer, log_line):
    # Encode the log line into tokens and move to model's device
    inputs = tokenizer.encode(log_line, return_tensors="pt").to(model.device)
    # Forward pass with labels (language modeling loss)
    outputs = model(inputs, labels=inputs)
    loss = outputs.loss
    perplexity = math.exp(loss.item())
    return perplexity

# Test with a sample log (using one of the parsed logs as example)
sample_log = parsed_logs[0]
pp = compute_perplexity(model, tokenizer, sample_log)
print(f"Perplexity for sample log: {pp:.2f}")


# %%
tokens, importance = extract_attention(model, tokenizer, sample_log)
for token, imp in zip(tokens, importance.tolist()):
    print(f"{token}: {imp:.3f}")

# %%
import torch
import matplotlib.pyplot as plt

def generate_explanation(model, tokenizer, log_line, top_k=3):
    """
    Generate a textual explanation for a log entry based on its attention scores.
    This function extracts tokens, normalizes their attention scores, and
    identifies the top-k tokens that most influenced the prediction.
    """
    tokens, importance = extract_attention(model, tokenizer, log_line)
    
    # Normalize attention scores so that they sum to 1 (for easier interpretation)
    norm_importance = importance / torch.sum(importance)
    
    # Retrieve the top-k tokens with the highest normalized attention scores
    top_values, top_indices = torch.topk(norm_importance, top_k)
    
    # Create a list of tuples (token, normalized score)
    explanation_tokens = [(tokens[i], top_values[i].item()) for i in top_indices]
    
    # Create a narrative explanation
    explanation_msg = (
        "The log entry was flagged as anomalous because the following tokens "
        "received the highest attention: " +
        ", ".join([f"'{tok}' (score: {score:.2f})" for tok, score in explanation_tokens]) +
        "."
    )
    return explanation_msg, explanation_tokens

def visualize_attention(tokens, importance):
    """
    Visualize the attention scores as a bar chart.
    """
    plt.figure(figsize=(10, 2))
    plt.bar(range(len(tokens)), importance.tolist(), tick_label=tokens)
    plt.xticks(rotation=90)
    plt.xlabel("Tokens")
    plt.ylabel("Attention Score")
    plt.title("Token Attention Weights")
    plt.tight_layout()
    plt.show()

# Example usage:
# Assume `model` and `tokenizer` are already loaded (e.g., GPT2LMHeadModel and GPT2Tokenizer),
# and `sample_log` is one of your parsed log lines.
sample_log = "ERROR: HDFC_v1 connection timeout after 500ms"  # Replace with your actual log string

# Generate explanation for the sample log entry
explanation, top_tokens = generate_explanation(model, tokenizer, sample_log, top_k=3)
print("Explanation:", explanation)
print("Top tokens with attention scores:", top_tokens)

# Optionally, visualize the attention for the entire log line
tokens, importance = extract_attention(model, tokenizer, sample_log)
visualize_attention(tokens, importance)

# %%
def detect_anomaly(log_line, threshold=50):
    perplexity = compute_perplexity(model, tokenizer, log_line)
    is_anomaly = perplexity > threshold
    return is_anomaly, perplexity

# Example usage:
is_anomaly, perplexity = detect_anomaly(sample_log, threshold=50)
print("Is Anomalous:", is_anomaly, "Perplexity:", perplexity)


# %%
def extract_attention(model, tokenizer, log_line):
    """
    Tokenize the input log line, run it through the model with attention output,
    and return the tokens along with aggregated attention scores.
    """
    inputs = tokenizer.encode(log_line, return_tensors="pt").to(model.device)
    outputs = model(inputs, output_attentions=True)
    
    # Stack and average attention across layers and heads
    all_attentions = torch.stack(outputs.attentions)  # (num_layers, batch, num_heads, seq_len, seq_len)
    avg_attention = torch.mean(all_attentions, dim=(0, 2))  # (batch, seq_len, seq_len)
    
    # Sum attention for each token over all positions
    token_importance = torch.sum(avg_attention[0], dim=0)
    
    # Convert token IDs back to text tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    return tokens, token_importance

def generate_explanation(model, tokenizer, log_line, top_k=3):
    """
    Generate a textual explanation for a log entry based on its attention scores.
    """
    tokens, importance = extract_attention(model, tokenizer, log_line)
    norm_importance = importance / torch.sum(importance)  # Normalize scores
    
    # Retrieve the top-k tokens with highest normalized attention scores
    top_values, top_indices = torch.topk(norm_importance, top_k)
    explanation_tokens = [(tokens[i], top_values[i].item()) for i in top_indices]
    
    explanation_msg = (
        "The log entry was flagged as anomalous because the following tokens "
        "received the highest attention: " +
        ", ".join([f"'{tok}' (score: {score:.2f})" for tok, score in explanation_tokens]) +
        "."
    )
    return explanation_msg, explanation_tokens

# Example usage:
explanation, top_tokens = generate_explanation(model, tokenizer, sample_log, top_k=3)
print("Explanation:", explanation)


# %%
def process_log_entry(log_line, threshold=50, top_k=3):
    # Step 1: Compute perplexity
    is_anomaly, perplexity = detect_anomaly(log_line, threshold)
    
    # Step 2: Generate explanation if anomaly is detected
    if is_anomaly:
        explanation, tokens_info = generate_explanation(model, tokenizer, log_line, top_k)
    else:
        explanation = "Log entry is normal."
        tokens_info = []
    
    return {
        "log_line": log_line,
        "perplexity": perplexity,
        "is_anomaly": is_anomaly,
        "explanation": explanation,
        "top_tokens": tokens_info
    }

# Process a sample log entry
result = process_log_entry(sample_log)
print(result)


# %%
# Compute token lengths for each tokenized log entry
token_lengths = [len(t) for t in tokenized_logs]

plt.figure(figsize=(8, 5))
plt.hist(token_lengths, bins=20, edgecolor="black")
plt.xlabel("Token Length")
plt.ylabel("Frequency")
plt.title("Token Length Distribution")
plt.tight_layout()
plt.show()


# %%
import numpy as np

# Compute perplexity for each log entry in raw_logs
perplexities = [compute_perplexity(model, tokenizer, log) for log in raw_logs]

plt.figure(figsize=(8, 5))
plt.hist(perplexities, bins=20, edgecolor="black")
plt.xlabel("Perplexity")
plt.ylabel("Frequency")
plt.title("Perplexity Distribution Across Log Lines")
plt.tight_layout()
plt.show()


