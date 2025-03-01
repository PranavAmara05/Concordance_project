import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
from torch.utils.data import Dataset
import sys
import subprocess

# Function to install a package
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}. Error: {e}")
    else:
        print(f"Successfully installed {package}")

# Install protobuf
install_package("protobuf")

# Load and Preprocess Dataset
def load_data():
    nouns_data = pd.read_excel("/home/k_manju/conc_new_LLM/Nouns.xlsx")
    verbs_data = pd.read_excel("/home/k_manju/conc_new_LLM/verbal.xlsx")

    data = pd.concat([nouns_data, verbs_data], ignore_index=True)
    data = data.dropna()
    data.columns = ['Inflected', 'Base']

    return data

# Define Custom Dataset Class
class SanskritDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=50):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = "inflect: " + row['Inflected']
        target_text = row['Base']

        input_ids = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        ).input_ids.squeeze()

        target_ids = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        ).input_ids.squeeze()

        return {
            'input_ids': input_ids,
            'labels': target_ids
        }

# Custom Evaluation Metrics
def compute_bleu(reference, prediction):
    ref_words = reference.split()
    pred_words = prediction.split()
    overlap = sum(min(ref_words.count(word), pred_words.count(word)) for word in set(pred_words))
    return overlap / len(ref_words) if ref_words else 0

def compute_rouge(reference, prediction):
    ref_words = set(reference.split())
    pred_words = set(prediction.split())
    intersection = ref_words & pred_words
    precision = len(intersection) / len(pred_words) if pred_words else 0
    recall = len(intersection) / len(ref_words) if ref_words else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

def compute_edit_distance(reference, prediction):
    dp = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        for j in range(len(prediction) + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif reference[i - 1] == prediction[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(reference)][len(prediction)]

def compute_accuracy(references, predictions):
    correct = sum(1 for ref, pred in zip(references, predictions) if ref.strip() == pred.strip())
    return correct / len(references) if references else 0

# Evaluate the Model
def evaluate_model(model, tokenizer, test_data):
    print("Evaluating model...")
    model.eval()
    predictions, references = [], []

    for _, row in test_data.iterrows():
        input_text = "inflect: " + row['Inflected']
        target_text = row['Base']

        input_ids = tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=50
        ).input_ids

        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=50)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(prediction)
        references.append(target_text)

    bleu_scores = [compute_bleu(ref, pred) for ref, pred in zip(references, predictions)]
    rouge_scores = [compute_rouge(ref, pred) for ref, pred in zip(references, predictions)]
    edit_distances = [compute_edit_distance(ref, pred) for ref, pred in zip(references, predictions)]
    accuracy = compute_accuracy(references, predictions)

    avg_bleu = np.mean(bleu_scores)
    avg_rouge = {
        'precision': np.mean([score['precision'] for score in rouge_scores]),
        'recall': np.mean([score['recall'] for score in rouge_scores]),
        'f1_score': np.mean([score['f1_score'] for score in rouge_scores])
    }
    avg_edit_distance = np.mean(edit_distances)

    print("Evaluation Metrics:")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average ROUGE Scores: {avg_rouge}")
    print(f"Average Edit Distance: {avg_edit_distance:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return predictions, references

# Main Function
def main():
    # Load data
    print("Loading data...")
    data = load_data()

    # Train-test split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Prepare tokenizer and model
    # Load T5-small model
    tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Prepare datasets
    # Train-validation-test split
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)  # 70% Train
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 15% Val, 15% Test

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    
    # Training arguments tuned for small dataset
    training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=3e-4,  # Increased learning rate for faster convergence
    per_device_train_batch_size=4,  # Smaller batch size to prevent overfitting
    per_device_eval_batch_size=4,
    num_train_epochs=10,  # More epochs to ensure learning on a small dataset
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    save_strategy="epoch",
    warmup_steps=100,  # Helps in better stability
    gradient_accumulation_steps=2,  # Helps when using small batches
    eval_steps=1,  # Evaluate at every epoch
    load_best_model_at_end=True  # Ensures the best model is saved
    )



    # Trainer setup
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Use validation set for evaluation
    tokenizer=tokenizer,
    data_collator=data_collator
    )


    # Train the model
    print("Training the model...")
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

    # Evaluate the model on both validation and test sets
    print("Evaluating on Validation Set:")
    val_predictions, val_references = evaluate_model(model, tokenizer, val_data)

    print("\nEvaluating on Test Set:")
    test_predictions, test_references = evaluate_model(model, tokenizer, test_data)

if __name__ == "__main__":
    main()
