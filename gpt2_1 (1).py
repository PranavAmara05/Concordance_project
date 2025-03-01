import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Read both Excel files into pandas DataFrames
df1 = pd.read_excel("/home/k_manju/conc_new_LLM/Nouns.xlsx")
df2 = pd.read_excel("/home/k_manju/conc_new_LLM/verbal.xlsx")

# Concatenate the two DataFrames along the rows (axis=0)
sanskrit_df = pd.concat([df1, df2], axis=0, ignore_index=True)

# Preparing data for training
sanskrit_df['input_text'] = sanskrit_df['Word'] + ' ->'
sanskrit_df['target_text'] = sanskrit_df['Base Word']

# Split the data into training and testing sets
train_df, test_df = train_test_split(sanskrit_df, test_size=0.2, random_state=42)

# Custom dataset class with chunk overlap enabled
class SanskritDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=50, chunk_overlap=5):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_overlap = chunk_overlap

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_text = self.dataframe.iloc[idx]['input_text']
        target_text = self.dataframe.iloc[idx]['target_text']

        # Combine input and target for GPT-2
        combined_text = input_text + " " + target_text

        # Tokenize combined text without truncation and with chunk overlap handling
        tokens = self.tokenizer(
            combined_text,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length + self.chunk_overlap  # Chunk overlap added
        )

        # Extract input_ids and attention_mask
        input_ids = tokens.input_ids.squeeze()
        attention_mask = tokens.attention_mask.squeeze()

        # Create labels: shift the input_ids for the output labels
        labels = input_ids.clone()
        input_token_length = len(self.tokenizer(input_text).input_ids)
        labels[:input_token_length] = -100  # We don't want to predict the input

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Load the GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add special tokens (for padding, if needed)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Create datasets
train_dataset = SanskritDataset(train_df, tokenizer)
test_dataset = SanskritDataset(test_df, tokenizer)

# Training arguments with increased epochs and additional configurations
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,  # Increased epochs
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_steps=10,
    save_total_limit=2,
    report_to='none',  # Use 'tensorboard' or 'wandb' if desired
    label_smoothing_factor=0.1,  # Helpful for language generation tasks
    learning_rate=5e-5,
    weight_decay=0.01
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(".\\sanskrit_gpt2_model")
tokenizer.save_pretrained(".\\sanskrit_gpt2_model")

# Function to predict the base word using the fine-tuned model
def predict_base_word_with_model(word):
    input_text = f"{word} ->"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=20)
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    base_word = predicted_text.split("->")[-1].strip()
    return base_word

# BLEU score calculation
def compute_bleu_score(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    matches = sum(1 for w in candidate_tokens if w in reference_tokens)
    return matches / len(candidate_tokens) if candidate_tokens else 0

# ROUGE-L score calculation
def compute_rouge_l(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    lcs_length = [[0] * (len(candidate_tokens) + 1) for _ in range(len(reference_tokens) + 1)]

    for i in range(1, len(reference_tokens) + 1):
        for j in range(1, len(candidate_tokens) + 1):
            if reference_tokens[i - 1] == candidate_tokens[j - 1]:
                lcs_length[i][j] = lcs_length[i - 1][j - 1] + 1
            else:
                lcs_length[i][j] = max(lcs_length[i - 1][j], lcs_length[i][j - 1])

    lcs = lcs_length[len(reference_tokens)][len(candidate_tokens)]
    precision = lcs / len(candidate_tokens) if candidate_tokens else 0
    recall = lcs / len(reference_tokens) if reference_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1

# Evaluation metrics
def compute_metrics(test_df):
    predictions, references = [], []
    for _, row in test_df.iterrows():
        word = row['Word']
        true_base_word = row['Base Word']
        predicted_base_word = predict_base_word_with_model(word)
        predictions.append(predicted_base_word)
        references.append(true_base_word)

    # Accuracy
    correct = sum([pred == ref for pred, ref in zip(predictions, references)])
    accuracy = correct / len(references)

    # Edit distance
    def edit_distance(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a) + 1):
            for j in range(len(b) + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[len(a)][len(b)]

    edit_distances = [edit_distance(pred, ref) for pred, ref in zip(predictions, references)]
    avg_edit_distance = sum(edit_distances) / len(edit_distances)

    # BLEU and ROUGE scores
    bleu_scores = [compute_bleu_score(ref, pred) for ref, pred in zip(references, predictions)]
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    rouge_scores = [compute_rouge_l(ref, pred) for ref, pred in zip(references, predictions)]
    avg_rouge_score = sum(rouge_scores) / len(rouge_scores)

    return accuracy, avg_edit_distance, avg_bleu_score, avg_rouge_score

# Evaluate the model and compute metrics
accuracy, avg_edit_distance, avg_bleu_score, avg_rouge_score = compute_metrics(test_df)
print(f"Model accuracy: {accuracy * 100:.2f}%")
print(f"Average edit distance: {avg_edit_distance:.2f}")
print(f"Average BLEU score: {avg_bleu_score:.2f}")
print(f"Average ROUGE-L score: {avg_rouge_score:.2f}")

# Plot evaluation results
def plot_metrics(accuracy, avg_edit_distance, avg_bleu_score, avg_rouge_score):
    metrics = ['Accuracy', 'Average Edit Distance', 'BLEU Score', 'ROUGE-L Score']
    values = [accuracy * 100, avg_edit_distance, avg_bleu_score * 100, avg_rouge_score * 100]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
    plt.title('Evaluation Metrics')
    plt.ylabel('Value')
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center', fontsize=12)
    plt.show()

plot_metrics(accuracy, avg_edit_distance, avg_bleu_score, avg_rouge_score)
