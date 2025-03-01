import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load dataset
nouns_data = pd.read_excel("/home/k_manju/conc_new_LLM/Nouns.xlsx")
verbs_data = pd.read_excel("/home/k_manju/conc_new_LLM/verbal.xlsx")

data = pd.concat([nouns_data, verbs_data], ignore_index=True)
data = data.dropna()  # Ensure no missing values
data.columns = ['Inflected', 'Base']  # Ensure correct column names

# Train-Validation-Test split
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)  # 70% train
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 15% val, 15% test

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
config = GPT2Config.from_pretrained('distilgpt2')

# Add a padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config.pad_token_id = tokenizer.pad_token_id

# Load model
model = GPT2LMHeadModel.from_pretrained('distilgpt2', config=config)
model.resize_token_embeddings(len(tokenizer))  # Adjust embeddings for new tokens

# Dataset class
class SanskritDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=50):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = row['Inflected']
        target_text = row['Base']

        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            target_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

# Prepare datasets
train_dataset = SanskritDataset(train_data, tokenizer)
val_dataset = SanskritDataset(val_data, tokenizer)
test_dataset = SanskritDataset(test_data, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",  # Evaluate after each epoch
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Using validation dataset
    tokenizer=tokenizer
)

# Train model
trainer.train()

# Save model and tokenizer
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

# Evaluate on test dataset
trainer.evaluate()

# Prediction and evaluation function
def evaluate_model(test_data, model, tokenizer):
    predictions, references = [], []
    bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, edit_distances, accuracies = [], [], [], [], [], []

    model.eval()
    for _, row in test_data.iterrows():
        input_text = row['Inflected']
        target_text = row['Base']
        input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=50)
        attention_mask = input_ids != tokenizer.pad_token_id  # Create attention mask

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,  # Generate up to 20 new tokens
                pad_token_id=tokenizer.pad_token_id
            )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(pred)
        references.append(target_text)

        # BLEU Score (Simplified Unigram Precision)
        pred_tokens = pred.split()
        ref_tokens = target_text.split()
        common_tokens = len(set(pred_tokens) & set(ref_tokens))
        precision = common_tokens / len(pred_tokens) if len(pred_tokens) > 0 else 0
        bleu_scores.append(precision)

        # ROUGE Scores
        overlap1 = len(set(pred_tokens[:1]) & set(ref_tokens[:1]))
        rouge1_scores.append(overlap1 / len(ref_tokens[:1]) if len(ref_tokens[:1]) > 0 else 0)
        overlap2 = len(set(pred_tokens[:2]) & set(ref_tokens[:2]))
        rouge2_scores.append(overlap2 / len(ref_tokens[:2]) if len(ref_tokens[:2]) > 0 else 0)
        overlapL = len(set(pred_tokens) & set(ref_tokens))
        rougeL_scores.append(overlapL / len(ref_tokens) if len(ref_tokens) > 0 else 0)

        # Edit Distance
        edit_distances.append(sum(1 for a, b in zip(pred, target_text) if a != b) + abs(len(pred) - len(target_text)))

        # Accuracy
        accuracies.append(1 if pred == target_text else 0)

    # Compute average metrics
    avg_bleu = np.mean(bleu_scores)
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    avg_rougeL = np.mean(rougeL_scores)
    avg_edit_distance = np.mean(edit_distances)
    avg_accuracy = np.mean(accuracies)

    print("Evaluation Metrics:")
    print(f"BLEU Score: {avg_bleu:.4f}")
    print(f"ROUGE-1 Score: {avg_rouge1:.4f}")
    print(f"ROUGE-2 Score: {avg_rouge2:.4f}")
    print(f"ROUGE-L Score: {avg_rougeL:.4f}")
    print(f"Average Edit Distance: {avg_edit_distance:.4f}")
    print(f"Accuracy: {avg_accuracy:.4f}")

    return predictions, references


# Plot results
def plot_results(metric_values, metric_name):
    plt.figure(figsize=(10, 5))
    plt.hist(metric_values, bins=20, alpha=0.7, label=metric_name)
    plt.xlabel(metric_name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {metric_name} Values')
    plt.legend()
    plt.show()


# Run evaluation for both validation and test datasets
print("Evaluating on Validation Set:")
val_predictions, val_references = evaluate_model(val_data, model, tokenizer)

print("\nEvaluating on Test Set:")
test_predictions, test_references = evaluate_model(test_data, model, tokenizer)

# Plot evaluation distributions for validation set
print("\nValidation Set Metrics:")
plot_results(bleu_scores, 'BLEU Scores (Validation)')
plot_results(rouge1_scores, 'ROUGE-1 Scores (Validation)')
plot_results(rouge2_scores, 'ROUGE-2 Scores (Validation)')
plot_results(rougeL_scores, 'ROUGE-L Scores (Validation)')
plot_results(edit_distances, 'Edit Distances (Validation)')
plot_results(accuracies, 'Accuracies (Validation)')

# Plot evaluation distributions for test set
print("\nTest Set Metrics:")
plot_results(bleu_scores, 'BLEU Scores (Test)')
plot_results(rouge1_scores, 'ROUGE-1 Scores (Test)')
plot_results(rouge2_scores, 'ROUGE-2 Scores (Test)')
plot_results(rougeL_scores, 'ROUGE-L Scores (Test)')
plot_results(edit_distances, 'Edit Distances (Test)')
plot_results(accuracies, 'Accuracies (Test)')
