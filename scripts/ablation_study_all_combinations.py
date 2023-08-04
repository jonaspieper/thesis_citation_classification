import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import random
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(42)

def encode_data(tokenizer, data, labels, max_length):
    input_ids = []
    attention_masks = []

    for text in data:
        encoded_dict = tokenizer.encode_plus(
                            text=text,
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


def train(max_length, epochs, feature_subset, train_df, test_df, val_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)


    fixed_feature_start = ['dynamic_context_next']
    fixed_feature_end = ['cited_abstract']
  
    target_column = 'citation_class_label'

    middle_features = list(feature_subset)

    features_start = fixed_feature_start + middle_features
    features_end = fixed_feature_end

    print(features_start)
    # Convert all the textual features to string type
    train_df[features_start + features_end] = train_df[features_start + features_end].astype(str)
    test_df[features_start + features_end] = test_df[features_start + features_end].astype(str)
    val_df[features_start + features_end] = val_df[features_start + features_end].astype(str)


    # Combine all the textual features into a single feature
    train_df['combined_text'] = train_df[features_start].apply(' [SEP] '.join, axis=1) + ' [SEP] ' + train_df[features_end[0]].astype(str)
    test_df['combined_text'] = test_df[features_start].apply(' [SEP] '.join, axis=1) + ' [SEP] ' + test_df[features_end[0]].astype(str)
    val_df['combined_text'] = val_df[features_start].apply(' [SEP] '.join, axis=1) + ' [SEP] ' + val_df[features_end[0]].astype(str)

    print(train_df.combined_text)
    # Get the maximum length of the combined_text for the tokenizer

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained(
        "allenai/scibert_scivocab_uncased", 
        num_labels=6, 
        output_attentions=False, 
        output_hidden_states=False,
    )

    # Define weights, criterion, optimizer
    weights = torch.tensor([0.29911462, 1.38427464, 3.01932367, 8.01282051, 1.8037518, 1.08225108]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr = 1e-5)
    
    # if you have multiple GPUs, let's use them
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    weights = weights.to(device)
    model = model.to(device)

    # Encode your data
    train_input_ids, train_attention_masks, train_labels = encode_data(tokenizer, train_df['combined_text'], train_df[target_column].values, max_length)
    val_input_ids, val_attention_masks, val_labels = encode_data(tokenizer, val_df['combined_text'], val_df[target_column].values, max_length)
    test_input_ids, test_attention_masks, test_labels = encode_data(tokenizer, test_df['combined_text'], test_df[target_column].values, max_length)

    # Create DataLoaders
    train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
    test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)

    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=4)
    validation_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=4)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=4)

    # Training loop
    for epoch_i in range(0, epochs):
        # Training
        model.train()

        # Loop over batches
        for step, batch in enumerate(train_dataloader):
            # Move batch tensors to the GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Compute loss
            loss = criterion(outputs[0], b_labels)
            loss.backward()

            # Update weights
            optimizer.step()

    # Evaluation
    model.eval()

    # Make sure the variables to store the evaluation results are on the same device as the model
    true_labels, predictions = [], []

    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    # Flatten outputs
    pred_flat = np.argmax(np.vstack(predictions), axis=1).flatten()
    labels_flat = np.concatenate(true_labels)

    # Calculate scores
    macro_f1 = f1_score(labels_flat, pred_flat, average='macro')
    micro_f1 = f1_score(labels_flat, pred_flat, average='micro')
    weighted_f1 = f1_score(labels_flat, pred_flat, average='weighted')
    precision = precision_score(labels_flat, pred_flat, average='weighted')
    recall = recall_score(labels_flat, pred_flat, average='weighted')

    return macro_f1, micro_f1, weighted_f1, precision, recall











def main():

    train_df = pd.read_csv('/home/roland/Projects/JP_citation_classification/dynamic_citation_extraction_original/data/non_contiguous_sdp_act_exp3/train.txt', sep="\t", engine="python", dtype=object)
    test_df = pd.read_csv('/home/roland/Projects/JP_citation_classification/dynamic_citation_extraction_original/data/non_contiguous_sdp_act_exp3/test.txt', sep="\t", engine="python", dtype=object)
    val_df = pd.read_csv('/home/roland/Projects/JP_citation_classification/dynamic_citation_extraction_original/data/non_contiguous_sdp_act_exp3/valid.txt', sep="\t", engine="python", dtype=object)






    add_data = pd.read_csv('/home/roland/Projects/JP_citation_classification/feature_scraping/data/enriched_data/7_added_causeminer_data.csv')

    extra_features = ['unique_id', 'cite_pos_in_sent', 'direct_citations', 'cited_publication_info', 'sent_pos_in_article', 'cited_concepts', 'diff_publication_date', 'citing_concepts']

    train_df = pd.merge(left = train_df, right = add_data[extra_features], on = 'unique_id', how = 'left')
    test_df = pd.merge(left = test_df, right = add_data[extra_features], on = 'unique_id', how = 'left')
    val_df = pd.merge(left = val_df, right = add_data[extra_features], on = 'unique_id', how = 'left')

    train_df.citation_class_label = train_df.citation_class_label.astype(int)
    test_df.citation_class_label = test_df.citation_class_label.astype(int)
    val_df.citation_class_label = val_df.citation_class_label.astype(int)

    # Fixed feature
    fixed_feature = ['dynamic_context_next']
    
    features = ['']

    #features = ['cited_concepts']

    # Generate all possible combinations of features for ablation study
    all_combinations = []
    for r in range(1, len(features) + 1):
        combinations_object = itertools.combinations(features, r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list
    
    print(all_combinations)

    # Create a dataframe to store results
    results = pd.DataFrame(columns=['features', 'macro_f1', 'micro_f1', 'weighted_f1', 'precision', 'recall'])


    for feature_subset in all_combinations:
        start_time = time.time()
        macro_f1, micro_f1, weighted_f1, precision, recall = train(512, 5, feature_subset, train_df, test_df, val_df)
        results = results.append({'features': feature_subset, 'macro_f1': macro_f1, 'micro_f1': micro_f1, 'weighted_f1': weighted_f1, 'precision': precision, 'recall': recall}, ignore_index=True)
        end_time = time.time()
        time_taken = end_time - start_time
        hours, remainder = divmod(time_taken, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_now = time.time()
        print(f"The Time now is {time_now}.")
        print(f"The run took {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds.")

    # Save the results
    results.to_csv('ablation_study_results.csv', index=False)

if __name__ == "__main__":
    main()

