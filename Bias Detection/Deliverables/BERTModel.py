# # from torch.utils.data import DataLoader, Dataset
# # import torch.nn as nn
# # from transformers import DistilBertModel
# # from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding 
# # from transformers import LongformerTokenizerFast, LongformerForSequenceClassification
# # from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# # from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
# # import torch
# # from transformers import get_linear_schedule_with_warmup
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # import os
# # from tqdm import tqdm
# # import logging
# # from sklearn.utils.class_weight import compute_class_weight
# # from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
# # from datetime import datetime
# # import random
# # import numpy as np
# # from sklearn.preprocessing import LabelEncoder
# # # Configure logging
# # # Load environment variables
# # from dotenv import load_dotenv
# # load_dotenv(dotenv_path='config.env', override=True)
# # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# # log_filename = f'bert_training_{timestamp}.log'
# # log_foldername = 'logs'
# # log_dir = f'/Users/amalkurian/Desktop/Dissertation/Bias Detection'
# # log_filepath = os.path.join(log_dir, log_foldername, log_filename)
# # os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
# # logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # article_df = pd.read_csv('/Users/amalkurian/Desktop/Dissertation/Bias Detection/Deliverables/Main_Dataset1.csv')
# # cur_dir= os.getcwd() #'/Users/amalkurian/Desktop/Dissertation/Bias Detection/Deliverables/'

# # os.chdir('..')
# # upl_dir= os.getcwd() #'/Users/amalkurian/Desktop/Dissertation/Bias Detection/models'

# # seed = 42
# # torch.manual_seed(seed)
# # random.seed(seed)
# # np.random.seed(seed)
# # torch.backends.cudnn.deterministic = True
# # torch.backends.cudnn.benchmark = False

# # le = LabelEncoder()
# # article_df['MATCH_LABELS_ENCODED'] = le.fit_transform(article_df['MATCH_LABELS'])
# # label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# # print(label_mapping)
# # # def chunk_text(text, tokenizer, max_length=512, stride = 64):
# # #     tokens = tokenizer.encode(text, add_special_tokens=True)
# # #     chunks = []
# # #     start = 0
# # #     while start < len(tokens):
# # #         end = start + max_length
# # #         chunk_tokens = tokens[start:end]
# # #         chunks.append(chunk_tokens)
# # #         if end >= len(tokens):
# # #             break
# # #         start += max_length - stride

# # #     chunk_dicts = []
# # #     for chunk in chunks:
# # #         encoded_chunk = tokenizer.prepare_for_model(
# # #             chunk,
# # #             max_length=max_length,
# # #             padding='max_length',
# # #             truncation=True,
# # #             return_tensors='pt')
# # #         chunk_dicts.append(encoded_chunk) # a list of dictionaries

# # #     return chunk_dicts

# # class CustomOmissionClassifier(nn.Module):
# #     def __init__(self, pretrained_model_name="allenai/longformer-base-4096", num_labels=2):
# #         super(CustomOmissionClassifier, self).__init__()
# #         self.omissionmodel = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name)
# #         # hidden_size = self.omissionmodel.config.hidden_size
# #         # self.classifier = nn.Linear(hidden_size, num_labels)
# #         self.loss_fn = nn.CrossEntropyLoss()


# #     def forward(self, input_ids, attention_mask=None, labels=None):
# #         outputs = self.omissionmodel(input_ids=input_ids, attention_mask=attention_mask)
# #         # last_hidden_state = outputs.last_hidden_state
# #         # pooled_output = last_hidden_state[:, 0]  # CLS token representation
# #         # logits = self.classifier(pooled_output)
# #         # Return a dict to mimic HF models
# #         return outputs.logits


# # model = CustomOmissionClassifier(num_labels=len(le.classes_))


# # # tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
# # # model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=2, trust_remote_code=True)


# # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# # # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


# # # tokenizer = BERTTokenizer.from_pretrained('bert-base-uncased')
# # # model = BERTForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # # tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")





# # class BiasDataset(Dataset):
# #     def __init__(self, texts, labels, tokenizer, max_length=512, stride=64):
# #         # Add domains into the intiailization for classification
# #         self.texts = texts
# #         self.labels = labels
# #         self.tokenizer = tokenizer
# #         self.max_length = max_length
# #         # self.stride = stride

# #         # self.samples = []
# #         # for text, label in zip(texts, labels):
# #         #     chunks = chunk_text(text, tokenizer, max_length, stride)
# #         #     for chunk in chunks:
# #         #         self.samples.append({
# #         #             'input_ids': chunk['input_ids'].squeeze(0),
# #         #             'attention_mask': chunk['attention_mask'].squeeze(0),
# #         #             'labels': torch.tensor(label, dtype=torch.long)
# #         #             })

# #     def __len__(self):
# #         return len(self.texts)

    
# #     def __getitem__(self, idx):
# #         # return self.samples[idx]
# #         text = self.texts[idx]
# #         label = self.labels[idx]
# #         encoding = self.tokenizer(
# #             text,
# #             truncation=True,
# #             add_special_tokens=True,
# #             return_attention_mask=True,
# #             padding=False,
# #             max_length=self.max_length,
# #             return_tensors='pt'
# #         )
# #         return {
# #             'input_ids': encoding['input_ids'].squeeze(0),
# #             'attention_mask': encoding['attention_mask'].squeeze(0),
# #             'labels': torch.tensor(label, dtype=torch.long)
# #         }
    

# # texts = (
# #     article_df['cleaned_content'].fillna('') + " "
# #     + article_df['entities_Group'].fillna('').astype(str) + " "
# #     + article_df['Actions'].fillna('').astype(str) + " "
# #     + article_df['Key_Phrases'].fillna('').astype(str)
# # ).tolist()

# # labels = article_df['MATCH_LABELS_ENCODED'].tolist()

# # #text_size = len(texts)*0.9
# # #train_data = texts[: text_size]
# # #validation_data = texts[text_size:]

# # train_data, validation_data, train_labels, validation_labels = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

# # train_labels_np = np.array(train_labels)

# # class_weights = compute_class_weight(
# #     class_weight='balanced',
# #     classes=np.unique(train_labels_np),
# #     y=train_labels_np
# # )

# # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# # train_dataset = BiasDataset(train_data, train_labels, tokenizer)
# # validation_dataset = BiasDataset(validation_data, validation_labels, tokenizer)

# # train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True,collate_fn=data_collator, num_workers=0)
# # validation_loader = DataLoader(validation_dataset, batch_size=5, shuffle=False, collate_fn=data_collator, num_workers = 0)


# # device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # weights = torch.tens
# # or(class_weights, dtype=torch.float).to(device)
# # print(f"Using class weights: {weights}")

# # # Define your loss function with these weights
# # loss_fn = nn.CrossEntropyLoss(weight=weights)

# # model.to(device)
# # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# # scheduler = get_linear_schedule_with_warmup(
# #     optimizer,
# #     num_warmup_steps=0,
# #     num_training_steps=len(train_loader) * 4  # Assuming 3 epochs
# # )

# # best_val_loss = float('inf')

# # for epoch in range(4):
# #     model.train()
# #     epoch_loss = 0.0
# #     for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
# #         batch = {k: v.to(device) for k, v in batch.items()}
# #         # Move the batch to the appropriate device
# #         optimizer.zero_grad()
# #         #input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
# #         #attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
# #         labels = batch['labels']
         
# #         logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
# #         train_loss = loss_fn(logits, labels)
# #         # loss_fn = torch.nn.CrossEntropyLoss()
# #         # train_loss = loss_fn(outputs.logits, labels)
# #         train_loss.backward()
# #         optimizer.step()
# #         scheduler.step()
# #         epoch_loss += train_loss.item()
# #     average_loss = epoch_loss / len(train_loader) # total loss / number of batches

# #     model.eval()
# #     correct_predictions = 0
# #     total_optim_loss = 0.0
# #     all_preds = []
# #     all_labels = []
# #     with torch.no_grad():    # Similar like training so no gradients required, turning it off helps save memory
# #         for batch in tqdm(validation_loader, desc="Validating"):
# #             batch = {k: v.to(device) for k, v in batch.items()}
# #             labels = batch['labels']
# #             #input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
# #             #attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
# #             #labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')

# #             logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
# #             val_loss = loss_fn(logits, labels)
# #             total_optim_loss += val_loss.item()

# #             predicted_labels = torch.argmax(logits, dim=1)

# #             # Count correct predictions
# #             all_preds.extend(predicted_labels.cpu().numpy())
# #             all_labels.extend(labels.cpu().numpy())
# #             correct_predictions += (predicted_labels == labels).sum().item()
        
# #     average_val_loss = total_optim_loss / len(validation_loader)
# #     validation_accuracy = correct_predictions / len(validation_dataset)

# #     precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
# #     cm = confusion_matrix(all_labels, all_preds)

# #     print(f'Validation Loss: {average_val_loss:.4f}, Accuracy: {validation_accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
# #     print(f'Confusion Matrix:\n{cm}')
# #     logging.info(f'Epoch {epoch + 1}, Loss: {average_val_loss:.4f}, Accuracy: {validation_accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
# #     logging.info(f'Confusion Matrix:\n{cm}')

# #     if average_loss < best_val_loss:
# #         best_val_loss = average_loss
# #         torch.save(model.state_dict(), '/Users/amalkurian/Desktop/Dissertation/Bias Detection/Deliverables/best_model.pth')
# #         logging.info(f'Saved best model with loss: {best_val_loss}')


# # # Save the final model
# # file_name = 'final_model.pth'
# # file_folder = 'final_model_folder'
# # full_folder_path = os.path.join(upl_dir, file_folder)
# # os.makedirs(full_folder_path, exist_ok=True)
# # final_model_path = os.path.join(full_folder_path, file_name)
# # torch.save(model.state_dict(), final_model_path)



# from torch.utils.data import DataLoader, Dataset
# import torch.nn as nn
# # from transformers import DistilBertModel
# from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
# from transformers import LongformerTokenizerFast, LongformerForSequenceClassification
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
# from torch.utils.data import DataLoader, Dataset
# import torch
# from transformers import get_linear_schedule_with_warmup
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os
# from tqdm import tqdm
# import logging
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
# from datetime import datetime
# import random
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# # Configure logging
# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv(dotenv_path='config.env', override=True)
# timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# log_filename = f'bert_training_{timestamp}.log'
# log_foldername = 'logs'
# log_dir = f'/Users/amalkurian/Desktop/Dissertation/Bias Detection'
# log_filepath = os.path.join(log_dir, log_foldername, log_filename)
# os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
# logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# article_df = pd.read_csv('/content/Main_Dataset3.csv')
# cur_dir= os.getcwd() #'/Users/amalkurian/Desktop/Dissertation/Bias Detection/Deliverables/'

# os.chdir('..')
# upl_dir= '/content/Deliverables' #'/Users/amalkurian/Desktop/Dissertation/Bias Detection/models'

# seed = 42
# torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# le = LabelEncoder()
# article_df['MATCH_LABELS_ENCODED'] = le.fit_transform(article_df['MATCH_LABELS'])
# label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# print(label_mapping)
# # def chunk_text(text, tokenizer, max_length=512, stride = 64):
# #     tokens = tokenizer.encode(text, add_special_tokens=True)
# #     chunks = []
# #     start = 0
# #     while start < len(tokens):
# #         end = start + max_length
# #         chunk_tokens = tokens[start:end]
# #         chunks.append(chunk_tokens)
# #         if end >= len(tokens):
# #             break
# #         start += max_length - stride

# #     chunk_dicts = []
# #     for chunk in chunks:
# #         encoded_chunk = tokenizer.prepare_for_model(
# #             chunk,
# #             max_length=max_length,
# #             padding='max_length',
# #             truncation=True,
# #             return_tensors='pt')
# #         chunk_dicts.append(encoded_chunk) # a list of dictionaries

# #     return chunk_dicts

# class CustomOmissionClassifier(nn.Module):
#     def __init__(self, pretrained_model_name="distilbert/distilbert-base-uncased", num_labels=2):
#         super(CustomOmissionClassifier, self).__init__()
#         config = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name).config
#         config.dropout = dropout
#         config.attention_dropout = attention_dropout
#         self.omissionmodel = DistilBertForSequenceClassification.from_pretrained(
#             pretrained_model_name,
#             num_labels=num_labels)
#         # hidden_size = self.omissionmodel.config.hidden_size
#         # self.classifier = nn.Linear(hidden_size, num_labels)
#         self.loss_fn = nn.CrossEntropyLoss()


#     def forward(self, input_ids, attention_mask=None, labels=None):
#         outputs = self.omissionmodel(input_ids=input_ids, attention_mask=attention_mask)
#         # last_hidden_state = outputs.last_hidden_state
#         # pooled_output = last_hidden_state[:, 0]  # CLS token representation
#         # logits = self.classifier(pooled_output)
#         # Return a dict to mimic HF models
#         return outputs.logits


# model = CustomOmissionClassifier(num_labels=len(le.classes_))


# # tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
# # model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=2, trust_remote_code=True)


# tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
# # model = DistilBertForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased', num_labels=2)


# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # model = BERTForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")


# class BiasDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=512, stride=64):
#         # Add domains into the intiailization for classification
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         # self.stride = stride

#         # self.samples = []
#         # for text, label in zip(texts, labels):
#         #     chunks = chunk_text(text, tokenizer, max_length, stride)
#         #     for chunk in chunks:
#         #         self.samples.append({
#         #             'input_ids': chunk['input_ids'].squeeze(0),
#         #             'attention_mask': chunk['attention_mask'].squeeze(0),
#         #             'labels': torch.tensor(label, dtype=torch.long)
#         #             })

#     def __len__(self):
#         return len(self.texts)


#     def __getitem__(self, idx):
#         # return self.samples[idx]
#         text = self.texts[idx]
#         label = self.labels[idx]
#         encoding = self.tokenizer(
#             text,
#             truncation=True,
#             add_special_tokens=True,
#             return_attention_mask=True,
#             padding=False,
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encoding['input_ids'].squeeze(0),
#             'attention_mask': encoding['attention_mask'].squeeze(0),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }


# texts = (
#     article_df['cleaned_content'].fillna('') + " "
#     + article_df['entities_Group'].fillna('').astype(str) + " "
#     + article_df['Actions'].fillna('').astype(str) + " "
#     + article_df['Key_Phrases'].fillna('').astype(str)
# ).tolist()

# labels = article_df['MATCH_LABELS_ENCODED'].tolist()

# #text_size = len(texts)*0.9
# #train_data = texts[: text_size]
# #validation_data = texts[text_size:]

# train_data, validation_data, train_labels, validation_labels = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

# train_labels_np = np.array(train_labels)

# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(train_labels_np),
#     y=train_labels_np
# )

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# train_dataset = BiasDataset(train_data, train_labels, tokenizer)
# validation_dataset = BiasDataset(validation_data, validation_labels, tokenizer)

# train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True,collate_fn=data_collator, num_workers=0)
# validation_loader = DataLoader(validation_dataset, batch_size=5, shuffle=False, collate_fn=data_collator, num_workers = 0)


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# weights = torch.tensor(class_weights, dtype=torch.float).to(device)
# print(f"Using class weights: {weights}")

# # Define your loss function with these weights
# loss_fn = nn.CrossEntropyLoss(weight=weights)

# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=0,
#     num_training_steps=len(train_loader) * 4  # Assuming 3 epochs
# )

# best_val_loss = float('inf')

# for epoch in range(4):
#     model.train()
#     epoch_loss = 0.0
#     for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         # Move the batch to the appropriate device
#         optimizer.zero_grad()
#         #input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
#         #attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
#         labels = batch['labels']

#         logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
#         train_loss = loss_fn(logits, labels)
#         # loss_fn = torch.nn.CrossEntropyLoss()
#         # train_loss = loss_fn(outputs.logits, labels)
#         train_loss.backward()
#         optimizer.step()
#         scheduler.step()
#         epoch_loss += train_loss.item()
#     average_loss = epoch_loss / len(train_loader) # total loss / number of batches

#     model.eval()
#     correct_predictions = 0
#     total_optim_loss = 0.0
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():    # Similar like training so no gradients required, turning it off helps save memory
#         for batch in tqdm(validation_loader, desc="Validating"):
#             batch = {k: v.to(device) for k, v in batch.items()}
#             labels = batch['labels']
#             #input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
#             #attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
#             #labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')

#             logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
#             val_loss = loss_fn(logits, labels)
#             total_optim_loss += val_loss.item()

#             predicted_labels = torch.argmax(logits, dim=1)

#             # Count correct predictions
#             all_preds.extend(predicted_labels.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             correct_predictions += (predicted_labels == labels).sum().item()

#     average_val_loss = total_optim_loss / len(validation_loader)
#     validation_accuracy = correct_predictions / len(validation_dataset)

#     precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
#     cm = confusion_matrix(all_labels, all_preds)

#     print(f'Validation Loss: {average_val_loss:.4f}, Accuracy: {validation_accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
#     print(f'Confusion Matrix:\n{cm}')
#     logging.info(f'Epoch {epoch + 1}, Loss: {average_val_loss:.4f}, Accuracy: {validation_accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
#     logging.info(f'Confusion Matrix:\n{cm}')

#     if average_loss < best_val_loss:
#         best_val_loss = average_loss
#         torch.save(model.state_dict(), '/content/Deliverables/best_model.pth')
#         logging.info(f'Saved best model with loss: {best_val_loss}')


# # Save the final model
# file_name = 'final_model.pth'
# file_folder = 'final_model_folder'
# full_folder_path = os.path.join(upl_dir, file_folder)
# os.makedirs(full_folder_path, exist_ok=True)
# final_model_path = os.path.join(full_folder_path, file_name)
# torch.save(model.state_dict(), final_model_path)








from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
# from transformers import DistilBertModel
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import logging
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from datetime import datetime
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Configure logging
# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path='/Users/amalkurian/Desktop/Dissertation/Bias Detection/env/config.env', override=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

log_filename = f'bert_training_{timestamp}.log'
log_foldername = 'logs'
log_dir = f'/Users/amalkurian/Desktop/Dissertation/Bias Detection'
log_filepath = os.path.join(log_dir, log_foldername, log_filename)
os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

article_df = pd.read_csv('/Users/amalkurian/Desktop/Dissertation/Bias Detection/Deliverables/Main_Dataset2.csv')
cur_dir= os.getcwd() #'/Users/amalkurian/Desktop/Dissertation/Bias Detection/Deliverables/'

os.chdir('..')
upl_dir= '/content/Deliverables' #'/Users/amalkurian/Desktop/Dissertation/Bias Detection/models'

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

le = LabelEncoder()
article_df['MATCH_LABELS_ENCODED'] = le.fit_transform(article_df['MATCH_LABELS'])
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(label_mapping)
# def chunk_text(text, tokenizer, max_length=512, stride = 64):
#     tokens = tokenizer.encode(text, add_special_tokens=True)
#     chunks = []
#     start = 0
#     while start < len(tokens):
#         end = start + max_length
#         chunk_tokens = tokens[start:end]
#         chunks.append(chunk_tokens)
#         if end >= len(tokens):
#             break
#         start += max_length - stride

#     chunk_dicts = []
#     for chunk in chunks:
#         encoded_chunk = tokenizer.prepare_for_model(
#             chunk,
#             max_length=max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt')
#         chunk_dicts.append(encoded_chunk) # a list of dictionaries

#     return chunk_dicts

class CustomOmissionClassifier(nn.Module):
    def __init__(self, pretrained_model_name="distilbert/distilbert-base-uncased", num_labels=2, dropout =0.3, attention_dropout=0.3):
        super(CustomOmissionClassifier, self).__init__()
        config = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name).config
        config.dropout = dropout
        config.attention_dropout = attention_dropout
        self.omissionmodel = DistilBertForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels)
        # hidden_size = self.omissionmodel.config.hidden_size
        # self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.omissionmodel(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state = outputs.last_hidden_state
        # pooled_output = last_hidden_state[:, 0]  # CLS token representation
        # logits = self.classifier(pooled_output)
        # Return a dict to mimic HF models
        return outputs.logits


model = CustomOmissionClassifier(num_labels=len(le.classes_))


# tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
# model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=2, trust_remote_code=True)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
# model = DistilBertForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased', num_labels=2)


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BERTForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")





class BiasDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, stride=64):
        # Add domains into the intiailization for classification
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.stride = stride

        # self.samples = []
        # for text, label in zip(texts, labels):
        #     chunks = chunk_text(text, tokenizer, max_length, stride)
        #     for chunk in chunks:
        #         self.samples.append({
        #             'input_ids': chunk['input_ids'].squeeze(0),
        #             'attention_mask': chunk['attention_mask'].squeeze(0),
        #             'labels': torch.tensor(label, dtype=torch.long)
        #             })

    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        # return self.samples[idx]
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


texts = (
    article_df['cleaned_content'].fillna('') + " "
    + article_df['entities_Group'].fillna('').astype(str) + " "
    + article_df['Actions'].fillna('').astype(str) + " "
    + article_df['Key_Phrases'].fillna('').astype(str)
).tolist()

labels = article_df['MATCH_LABELS_ENCODED'].tolist()

#text_size = len(texts)*0.9
#train_data = texts[: text_size]
#validation_data = texts[text_size:]

train_data, validation_data, train_labels, validation_labels = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

train_labels_np = np.array(train_labels)

class_weights = compute_class_weight(
    class_weight='balanced', # calculate weights, inversely proportional to class freq
    classes=np.unique(train_labels_np), # class mentioned in the train labels
    y=train_labels_np # count the class freq
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = BiasDataset(train_data, train_labels, tokenizer)
validation_dataset = BiasDataset(validation_data, validation_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True,collate_fn=data_collator, num_workers=0)
validation_loader = DataLoader(validation_dataset, batch_size=5, shuffle=True, collate_fn=data_collator, num_workers = 0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Using class weights: {weights}") # converting torch.tensor -> float

# Define your loss function with these weights
loss_fn = nn.CrossEntropyLoss(weight=weights)

model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * 4  # Assuming 3 epochs
)

best_val_loss = float('inf')

for epoch in range(4):
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        # Move the batch to the appropriate device
        optimizer.zero_grad() # clearing the previous gradients for memory
        #input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
        #attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = batch['labels']

        logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        train_loss = loss_fn(logits, labels)
        # loss_fn = torch.nn.CrossEntropyLoss()
        # train_loss = loss_fn(outputs.logits, labels)
        train_loss.backward() #back propagation
        optimizer.step()  # adjusts stored gradients as a part of learning
        scheduler.step() # increases learning rate each epoch
        epoch_loss += train_loss.item()
    average_loss = epoch_loss / len(train_loader) # total loss / number of batches = 

    model.eval()
    correct_predictions = 0
    total_optim_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():    # Similar like training so no gradients required, turning it off helps save memory
        for batch in tqdm(validation_loader, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']
            #input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            #attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            #labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')

            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            val_loss = loss_fn(logits, labels)
            total_optim_loss += val_loss.item()

            predicted_labels = torch.argmax(logits, dim=1)

            # Count correct predictions
            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct_predictions += (predicted_labels == labels).sum().item()

    average_val_loss = total_optim_loss / len(validation_loader)
    validation_accuracy = correct_predictions / len(validation_dataset)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    print(f'Validation Loss: {average_val_loss:.4f}, Accuracy: {validation_accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    print(f'Confusion Matrix:\n{cm}')
    logging.info(f'Epoch {epoch + 1}, Loss: {average_val_loss:.4f}, Accuracy: {validation_accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    logging.info(f'Confusion Matrix:\n{cm}')

    if average_loss < best_val_loss:
        best_val_loss = average_loss
        torch.save(model.state_dict(), '/Users/amalkurian/Desktop/Dissertation/Bias Detection/models/DistilBERT/best_model.pth')
        logging.info(f'Saved best model with loss: {best_val_loss}')


# Save the final model
file_name = 'final_model.pth'
file_folder = 'final_model_folder'
full_folder_path = os.path.join(upl_dir, file_folder)
os.makedirs(full_folder_path, exist_ok=True)
final_model_path = os.path.join(full_folder_path, file_name)
torch.save(model.state_dict(), final_model_path)







