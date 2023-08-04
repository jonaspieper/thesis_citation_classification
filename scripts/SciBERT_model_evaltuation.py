import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def encode_data(tokenizer, data, labels, max_length):
    input_ids = []
    attention_masks = []

    text_columns = [col for col in data.columns if col.startswith('text')]

    for _, row in data.iterrows():
        text = ' [SEP] '.join(row[col] for col in text_columns)
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

def main():
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)

    train_df = pd.read_csv('/home/roland/Projects/JP_citation_classification/dynamic_citation_extraction_original/data/fixed_context_sdp_act_mod/train.csv')#, sep="\t", engine="python", dtype=object
    test_df = pd.read_csv('/home/roland/Projects/JP_citation_classification/dynamic_citation_extraction_original/data/fixed_context_sdp_act_mod/test.csv')
    val_df = pd.read_csv('/home/roland/Projects/JP_citation_classification/dynamic_citation_extraction_original/data/fixed_context_sdp_act_mod/val.csv')

    add_data = pd.read_csv('/home/roland/Projects/JP_citation_classification/feature_scraping/data/enriched_data/7_added_causeminer_data.csv')

    extra_features = ['unique_id', 'cite_pos_in_sent', 'direct_citations', 'cited_publication_info', 'sent_pos_in_article', 'cited_concepts', 'diff_publication_date', 'citing_concepts']

    train_df = pd.merge(left = train_df, right = add_data[extra_features], on = 'unique_id', how = 'left')
    test_df = pd.merge(left = test_df, right = add_data[extra_features], on = 'unique_id', how = 'left')
    val_df = pd.merge(left = val_df, right = add_data[extra_features], on = 'unique_id', how = 'left')

    train_df.citation_class_label = train_df.citation_class_label.astype(int)
    test_df.citation_class_label = test_df.citation_class_label.astype(int)
    val_df.citation_class_label = val_df.citation_class_label.astype(int)
    #, 'direct_citations', 'sent_position_in_article', 'citing_concepts'
    text_columns = ['citation_context']
    target_column = 'citation_class_label'
    print(text_columns)
    train_df[text_columns] = train_df[text_columns].astype(str)
    test_df[text_columns] = test_df[text_columns].astype(str)
    val_df[text_columns] = val_df[text_columns].astype(str)

    new_df = train_df[text_columns].apply(' [SEP] '.join, axis=1).to_frame('combined_text')

    max_length = max([len(tokenizer.encode(text)) for text in new_df.combined_text])

    dataframes = {'train': train_df, 'test': test_df, 'val': val_df}

    for name in dataframes.keys():
        dataframes[name] = dataframes[name][text_columns + [target_column]].copy()
    
        new_column_names = {col: f'text_{col}' for col in text_columns}
        new_column_names[target_column] = f'target_{target_column}'

        dataframes[name].rename(columns=new_column_names, inplace=True)
    
    train_df, test_df, val_df = dataframes['train'], dataframes['test'], dataframes['val']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #[0.30435841, 1.34843581, 2.91375291, 7.57575758, 1.78062678, 1.06837607]
    weights = torch.tensor([0.29911462, 1.38427464, 3.01932367, 8.01282051, 1.8037518, 1.08225108]).to(device)
    dropout = 0.2
    learning_rate = 1e-5
    batch_size = 4
    epochs = 5
    num_labels = 6

    model = AutoModelForSequenceClassification.from_pretrained(
        "allenai/scibert_scivocab_uncased", 
        num_labels = num_labels, 
        output_attentions = False,
        output_hidden_states = False, 
    )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    weights = weights.to(device)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr = learning_rate)

    target_column = [col for col in train_df.columns if 'target_' in col][0]

    train_input_ids, train_attention_masks, train_labels = encode_data(tokenizer, train_df, train_df[target_column].values, 512)
    val_input_ids, val_attention_masks, val_labels = encode_data(tokenizer, val_df, val_df[target_column].values, 512)
    test_input_ids, test_attention_masks, test_labels = encode_data(tokenizer, test_df, test_df[target_column].values, 512)

    train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
    test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)

    batch_size = 4
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
    validation_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=batch_size)

    for epoch_i in range(0, epochs):
        total_train_loss = 0
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
        
            model.zero_grad()        
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        
            loss = torch.mean(outputs.loss)
            logits = outputs.logits

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0

        for batch in tqdm(validation_dataloader, total=len(validation_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():        
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = torch.mean(outputs.loss)
            logits = outputs.logits

            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_dataloader)
        print("Training and Validation done!")

    model.eval()

    predictions , true_labels = [], []

    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.extend(np.argmax(logits, axis=1).flatten())
        true_labels.extend(label_ids.flatten())

    predictions_np = np.array(predictions)
    true_labels_np = np.array(true_labels)

    # Save to CSV
    np.savetxt("results/fixed_pred.csv", predictions_np, delimiter=",", fmt='%d')
    np.savetxt("results/fixed_true_labels.csv", true_labels_np, delimiter=",", fmt='%d')

    f1_mac = f1_score(true_labels, predictions, average='macro')
    f1_mic = f1_score(true_labels, predictions, average='micro')
    f1_wei = f1_score(true_labels, predictions, average='weighted')


    print(f'Macro F1 Score: {f1_mac}')
    print(f'Micro F1 Score: {f1_mic}')
    print(f'Weighted F1 Score: {f1_wei}')

if __name__ == "__main__":
    main()
