import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, Trainer, TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import csv
import optuna
from torch.utils.data import Dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Micro F1
    f1_micro = f1_score(labels, preds, average='micro')
    
    # Macro F1
    f1_macro = f1_score(labels, preds, average='macro')

    # Weighted F1
    f1_weighted = f1_score(labels, preds, average='weighted')
    
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)

train_df = pd.read_csv('/home/roland/Projects/JP_citation_classification/dynamic_citation_extraction_original/data/fixed_context_sdp_act_mod/train.csv')
val_df = pd.read_csv('/home/roland/Projects/JP_citation_classification/dynamic_citation_extraction_original/data/fixed_context_sdp_act_mod/val.csv')
test_df = pd.read_csv('/home/roland/Projects/JP_citation_classification/dynamic_citation_extraction_original/data/fixed_context_sdp_act_mod/test.csv')

text_columns = ['cite_context_sent_+2']
target_column = 'citation_class_label'

train_df[text_columns] = train_df[text_columns].astype(str)
val_df[text_columns] = val_df[text_columns].astype(str)
test_df[text_columns] = test_df[text_columns].astype(str)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_labels = 6

model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    num_labels = num_labels,
    output_attentions = False,
    output_hidden_states = False,
)

model.to(device)
class_weights = torch.tensor([0.30435841, 1.34843581, 2.91375291, 7.57575758, 1.78062678, 1.06837607]).to(device)

def encode_data(tokenizer, data, labels, max_length):
    input_ids = []
    attention_masks = []
    
    for row in data:
        encoded_dict = tokenizer.encode_plus(
                            text=row,
                            add_special_tokens=True,
                            max_length=max_length,
                            pad_to_max_length=True,
                            return_attention_mask=True,
                            return_tensors='pt',
                        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

train_input_ids, train_attention_masks, train_labels = encode_data(tokenizer, train_df[text_columns[0]], train_df[target_column].values, 512)
val_input_ids, val_attention_masks, val_labels = encode_data(tokenizer, val_df[text_columns[0]], val_df[target_column].values, 512)
test_input_ids, test_attention_masks, test_labels = encode_data(tokenizer, test_df[text_columns[0]], test_df[target_column].values, 512)

train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        "allenai/scibert_scivocab_uncased", 
        num_labels = num_labels, 
        output_attentions = False,
        output_hidden_states = False,
    )

def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

def collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels),
    }


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

# Launch hyperparameter search using Optuna
trainer.hyperparameter_search(direction="maximize", backend="optuna", n_trials=2, compute_objective=None)


best_run = trainer.hyperparameter_search_results["best_run"]
print(f'Best trial id: {best_run.run_id}, Score: {best_run.objective}')
print(f'Best hyperparameters: {best_run.hyperparameters}')

# Save the best hyperparameters and all trials to csv files
with open('best_hyperparameters.csv', 'w') as f:
    writer = csv.writer(f)
    for key, value in best_run.hyperparameters.items():
        writer.writerow([key, value])

all_trials = [{'Trial ID': t.run_id, 'Score': t.objective, 'Hyperparameters': t.hyperparameters} for t in trainer.hyperparameter_search_results["trials"]]
df_all_trials = pd.DataFrame(all_trials)
df_all_trials.to_csv('all_trials.csv', index=False)