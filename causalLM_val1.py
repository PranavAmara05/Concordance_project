import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load dataset
print("Loading data...")

nouns_data = pd.read_excel("/home/k_manju/conc_new_LLM/Nouns.xlsx")
verbs_data = pd.read_excel("/home/k_manju/conc_new_LLM/verbal.xlsx")

data = pd.concat([nouns_data, verbs_data], ignore_index=True)
data = data.dropna()
data.columns = ['Inflected', 'Base']

# Train-test split
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Prepare tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
tokenizer.pad_token = tokenizer.eos_token
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')

# Dataset class
class SanskritDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=50):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = f"Inflected: {row['Inflected']}\nBase: "
        target_text = row['Base']
        
        full_text = input_text + target_text
        inputs = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        labels = inputs.input_ids.clone()
        labels[inputs.input_ids == tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.squeeze()
        }

train_dataset = SanskritDataset(train_data, tokenizer)
test_dataset = SanskritDataset(test_data, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=100,
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.05,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)


# Train the model
print("Training the model...")
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# Evaluation metrics functions
def calculate_bleu(reference, prediction):
    reference_words = reference.split()
    prediction_words = prediction.split()
    match_count = 0

    for word in prediction_words:
        if word in reference_words:
            match_count += 1

    precision = match_count / len(prediction_words) if prediction_words else 0
    brevity_penalty = min(1.0, len(prediction_words) / len(reference_words)) if reference_words else 0

    return precision * brevity_penalty

def calculate_rouge(reference, prediction):
    reference_words = set(reference.split())
    prediction_words = set(prediction.split())

    overlap = reference_words.intersection(prediction_words)
    recall = len(overlap) / len(reference_words) if reference_words else 0
    precision = len(overlap) / len(prediction_words) if prediction_words else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return recall, precision, f1_score

def calculate_edit_distance(reference, prediction):
    m, n = len(reference), len(prediction)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif reference[i - 1] == prediction[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

def calculate_accuracy(references, predictions):
    correct = sum(1 for ref, pred in zip(references, predictions) if ref.strip() == pred.strip())
    return correct / len(references) if references else 0

# Evaluation function
def evaluate_model(test_data, model, tokenizer):
    predictions, references = [], []
    bleu_scores, rouge_scores, edit_distances = [], [], []

    print("Evaluating...")
    model.eval()

    for _, row in test_data.iterrows():
        input_text = f"Inflected: {row['Inflected']}\nBase: "
        target_text = row['Base']

        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        )
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        with torch.no_grad():
            outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            eos_token_id=tokenizer.eos_token_id
        )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Base:")[-1].strip()

        predictions.append(pred)
        references.append(target_text)

        # Calculate metrics
        bleu_scores.append(calculate_bleu(target_text, pred))
        rouge = calculate_rouge(target_text, pred)
        rouge_scores.append(rouge)
        edit_distances.append(calculate_edit_distance(target_text, pred))

    avg_bleu = np.mean(bleu_scores)
    avg_rouge = np.mean([r[2] for r in rouge_scores])  # F1-score
    avg_edit_distance = np.mean(edit_distances)
    accuracy = calculate_accuracy(references, predictions)

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"BLEU Score: {avg_bleu:.4f}")
    print(f"ROUGE F1-Score: {avg_rouge:.4f}")
    print(f"Average Edit Distance: {avg_edit_distance:.4f}")

        # Save all predictions and references in a DataFrame
    results_df = pd.DataFrame({
        "Inflected": test_data["Inflected"].values,
        "Actual_Base": references,
        "Predicted_Base": predictions,
        "BLEU_Score": bleu_scores,
        "ROUGE_F1": [r[2] for r in rouge_scores],
        "Edit_Distance": edit_distances
    })

    return predictions, references, results_df


# Run evaluation
# Evaluate on validation set
print("\nEvaluating on validation dataset...")
val_predictions, val_references, val_df = evaluate_model(val_data, model, tokenizer)
val_df.to_csv("validation_results.csv", index=False)

# Evaluate on test set
print("\nEvaluating on test dataset...")
test_predictions, test_references, test_df = evaluate_model(test_data, model, tokenizer)
test_df.to_csv("test_results.csv", index=False)
