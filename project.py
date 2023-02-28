import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
import time
import numpy as np
import sys
import argparse
import os
import torchtext
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix,  ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from dataclasses import fields
from torchtext import data as torchdata
import re
import csv

class FeedForwardNN(torch.nn.Module):
    
    def __init__(self, input_size, vocab_size, embedding_size):
        """Construct a simple MLP discriminator"""
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        
        # Number of input features is 12.
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.layer_1 = nn.Linear(input_size * embedding_size, 1024) 
        self.layer_2 = nn.Linear(1024, 512)
        self.layer_3 = nn.Linear(512, 256)
        self.layer_4 = nn.Linear(256, 128)
        self.layer_5 = nn.Linear(128,64)
        self.layer_6 = nn.Linear(64,16)
        self.layer_out = nn.Linear(16, 1) 
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        x = self.embeddings(inputs).view((-1,self.input_size * self.embedding_size))
        x = self.tanh(self.layer_1(x))
        x = self.tanh(self.layer_2(x))
        x = self.batchnorm1(x)
        x = self.tanh(self.layer_3(x))
        x = self.tanh(self.layer_4(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.tanh(self.layer_5(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_6(x))
        x = self.layer_out(x)
        return self.sigmoid(x)
    
class LSTMnet(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
    super(LSTMnet, self).__init__()
    self.hidden_dim = hidden_dim
    self.embedding = nn.Embedding(vocab_size,embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    embed=self.embedding(x)
    output, (h_n, c_n) = self.lstm(embed)
    out = self.dropout(output)
    out = self.fc(output[:,-1,:])
    # out = self.dropout(output)
    out = torch.sigmoid(out)
    return out

# Read from training files, update encodings
def read_encode_custom(file_names, threshold):
    vocab = []
    words = {}
    wID = 0
    for fn in file_names:
      with open(fn,'rt', encoding="utf8") as f:
          for line in f:
              line = line.replace('\n','')
              tokens = line.split(' ')
              for t in tokens:
                  if t == "":
                    continue
                  try:
                      elem = words[t]
                  except:
                      elem = [wID,0]
                      vocab.append(t)
                      wID = wID + 1
                  elem[1] = elem[1] + 1
                  words[t] = elem

    temp = words
    words = {}
    vocab = []
    lookup = {}
    wID = 0
    words['<unk>'] = [wID,100]
    lookup[wID] = '<unk>'
    vocab.append('<unk>')
    for t in temp:
        if temp[t][1] >= threshold:
            vocab.append(t)
            wID = wID + 1
            words[t] = [wID,temp[t][1]]
            lookup[wID] = t
    wID += 1
    words['<pad>'] = [wID, 100]
    vocab.append('<pad>')
    lookup[wID] = '<pad>'
    return [vocab,words,lookup]

def decode(lookup,corpus):
    
    text = ''
    for i in range(len(corpus)):
        wID = corpus[i]
        text = text + lookup[wID] + ' '
    return(text)

def encode(words,text):
    corpus = []
    tokens = text.split(' ')
    for t in tokens:
        if t == "":
          continue
        try:
            wID = words[t][0]
        except:
            wID = words['<unk>'][0]
        corpus.append(wID)
    return(corpus)

def encode_file(words, filename):
    corpus = []
    with open(filename,'rt', encoding="utf8") as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(' ')
            for t in tokens:
                if t == "":
                  continue
                try:
                    wID = words[t][0]
                except:
                    wID = words['<unk>'][0]
                corpus.append(wID)
    return corpus

# Parse a real/fake corpus, return lists of list of content between start token and end token
def parse_corpus(source, start_token, end_token):
  res = []
  new_line = []
  i = 0
  while i < len(source):
    if i + 2 <= len(source) and source[i+1] == start_token:
      new_line = []
      i += 3
    elif i + 2 <= len(source) and source[i+1] == end_token:
      res.append(new_line)
      i += 3
    else:
      new_line.append(source[i])
      i += 1
  return res

# Parse a mix corpus, return lists of list of content between start token and end token, and return their corresponding labels
def parse_mix_corpus(source, start_token, end_token, real_token):
  texts = []
  labels = []
  new_line = []
  i = 0
  while i < len(source):
    if i + 2 <= len(source) and source[i+1] == start_token:
      new_line = []
      i += 3
    elif i + 2 <= len(source) and source[i+1] == end_token:
      texts.append(new_line)
      labels.append(source[i+3]==real_token)
      i += 4
    else:
      new_line.append(source[i])
      i += 1
  return texts, labels

def get_correct_num(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item()

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

class FakeDetectDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = self.encodings[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def train_FFNN(train_loader, val_loader, test_loader, params):
    batch_size = params['batch_size']
    input_size = params['pad_size']
    epochs = params['epochs']
    print("Start Training", epochs)
    lr = params['lr']
    embedding_size = params['embedding_size']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FFNN = FeedForwardNN(input_size=input_size, vocab_size=params['vocab_size'], embedding_size=embedding_size).to(device).double()
    FFNN.apply(weights_init)
    loss_func = torch.nn.BCELoss().double()
    torch.manual_seed(0)
    opt_FFNN = torch.optim.SGD(FFNN.parameters(), lr=lr) 
    loss_train = np.zeros(epochs)
    loss_val = np.zeros(epochs)
    perplexity_val = np.zeros(epochs)
    perplexity_train = np.zeros(epochs)
    acc_val = np.zeros(epochs)

    best_model_val_acc = 0.0
    for epoch in range(epochs):
        # total_epoch_size = 0
        FFNN.zero_grad()
        ### Train
        for batch_idx, batch_data in enumerate(train_loader):
        
            FFNN.zero_grad()
            x = batch_data['input_ids'].to(device)
            n_batch = x.shape[0]
            y = batch_data['labels'].to(device).double()
            preds = FFNN(x).squeeze()
            l = loss_func(preds, y)
            l.backward()
            opt_FFNN.step()
            loss_train[epoch] += l.detach().item() * n_batch
            # total_epoch_size += n_batch
            loss_train[epoch] /= len(train_loader.dataset)
            ### Evlaluate
            # total_epoch_size = 0
            FFNN.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                x = batch_data['input_ids'].to(device)
                n_batch = x.shape[0]
                y = batch_data['labels'].to(device).double()
                preds = FFNN(x).squeeze()
                val_l = loss_func(preds, y)
                loss_val[epoch] += val_l.detach().item() * n_batch
                # total_epoch_size += n_batch

                ### cal acc
                predicted_correct = get_correct_num(y, preds)
                acc_val[epoch] += predicted_correct
        
        acc_val[epoch] /= len(val_loader.dataset)
        loss_val[epoch] /= len(val_loader.dataset)
        perplexity_val[epoch] = np.exp(loss_val[epoch])
        perplexity_train[epoch] = np.exp(loss_train[epoch])
        if acc_val[epoch] > best_model_val_acc:
            torch.save(FFNN.state_dict(), "ffnn_bestmodel.pt")
        print(f"Epoch: {epoch}", f"Train Loss: {loss_train[epoch]}", f"Val Loss: {loss_val[epoch]}", f"Val ACC: {acc_val[epoch]}", f"Val Perplexity: {perplexity_val[epoch]}")

def test_FFNN(test_loader, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_test= np.zeros(params['epochs'])
    predicted_correct = 0
    # FFNN.eval()
    best_model = FeedForwardNN(input_size=params['pad_size'], vocab_size=params['vocab_size'], embedding_size=params['embedding_size']).to(device).double()
    best_model.load_state_dict(torch.load("ffnn_bestmodel.pt"))
    best_model.eval()
    test_pred_results = []
    gt_label = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            x = batch_data['input_ids'].to(device)
            n_batch = x.shape[0]
            y = batch_data['labels'].to(device).double()
            preds = best_model(x).squeeze()
            y_pred = (preds > 0.5)
            test_pred_results += y_pred.cpu().tolist()
            gt_label += y.cpu().tolist()
            ### cal acc
            predicted_correct += get_correct_num(y, preds)

    print('Accuracy of test dataset: {}'.format(predicted_correct/len(test_loader.dataset)))

def classify_FFNN(params, start_token, end_token, pad_token, words):
    blind = encode_file(words, "blind.test.tok")
    blind_parse = parse_corpus(blind, start_token, end_token)
    blind_tensor = [torch.zeros(params['pad_size'])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in blind_parse:
        blind_tensor.append(torch.Tensor(s))
    blind_tensor = torch.nn.utils.rnn.pad_sequence(blind_tensor, True, pad_token).type(torch.long)[1:]
    blind_dataset = FakeDetectDataset(blind_tensor, torch.zeros(blind_tensor.shape[0]))
    blind_loader = DataLoader(blind_dataset, batch_size=1, shuffle=False)
    best_model = FeedForwardNN(input_size=params['pad_size'], vocab_size=params['vocab_size'], embedding_size=params['embedding_size']).to(device).double()
    best_model.load_state_dict(torch.load("ffnn_bestmodel.pt"))
    best_model.eval()
    blind_pred_results = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(blind_loader):
            x = batch_data['input_ids'].to(device)
            n_batch = x.shape[0]
            y = batch_data['labels'].to(device).double()
            preds = best_model(x).squeeze()
            y_pred = (preds > 0.5)
            blind_pred_results.append(y_pred.item())
    with open('ffnn_blind.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for label in blind_pred_results:
            curr_row = []
            if label:
                curr_row.append("[REAL]")
            else:
                curr_row.append("[FAKE]")
            writer.writerow(curr_row)

def train(model, train_loader, valid_loader, criterion, optimizer, num_epoch=50):
    train_loss_lst=[]
    valid_loss_lst=[]
    perplexity_list_train=[]
    perplexity_list_valid=[]

    for epoch in range(num_epoch):
        print("-------------------")
        print('Epoch {}/{}'.format(epoch, num_epoch-1))

        # Train
        train_loss = 0.0
        train_num = 0
        model.train()
        for step, batch in enumerate(train_loader):
            text = batch.data[0]
            label = batch.label.view(-1,1)
            label = label
            out = model(text)
            loss = criterion(out, label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(label)
            train_num += len(label)

        train_loss_lst.append(train_loss/train_num)
        print("Train loss: {:.4f}".format(train_loss/train_num))
        perplexity_list_train.append(math.exp(train_loss/train_num))
        print("The train perplexity is:   {}".format(math.exp(train_loss/train_num)))

        # Validation
        valid_loss = 0.0
        valid_num = 0
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                text = batch.data[0]
                label = batch.label.view(-1,1)
                label = label
                out = model(text)
                loss = criterion(out, label.float())
                valid_loss += loss.item() * len(label)
                valid_num += len(label)

        valid_loss_lst.append(valid_loss/valid_num)
        print("Validation loss: {:.4f}".format(valid_loss/valid_num))
        perplexity_list_valid.append(math.exp(valid_loss/valid_num))
        print("The valid perplexity is:   {}".format(math.exp(valid_loss/valid_num)))
        
    return model, train_loss_lst, valid_loss_lst, perplexity_list_train, perplexity_list_valid
    
def preprocess_data_LSTM():
   # Data Preprocessing    
  df_blind = pd.DataFrame(columns=['data', 'label'])

  with open("blind.test.tok",'rt', encoding="utf-8") as f:
    data = ""
    for line in f:
      line = line.replace('\n','')
      if line == "< start_bio >":
        continue
      elif line == "< end_bio >":
        data = re.sub(r"[^\w]+", " ", data)
        data = re.sub(r"[\d]+", "<NUM>", data)    
        df_blind.loc[len(df_blind.index)] = [data , 1]
        data = ""
      else:
        data = data + line + " "

  df_blind.to_csv("df_blind.csv")

  df_train = pd.DataFrame(columns=['data', 'label'])

  with open("real.train.tok",'rt', encoding="utf-8") as f:
    data = ""
    for line in f:
        line = line.replace('\n','')
        if line == "< start_bio >":
            continue
        elif line == "< end_bio >":
            data = re.sub(r"[^\w]+", " ", data)
            data = re.sub(r"[\d]+", "<NUM>", data)    
            df_train.loc[len(df_train.index)] = [data , 1]
            data = ""
        else:
            data = data + line + " "

  with open("fake.train.tok",'rt', encoding="utf-8") as f:
    data = ""
    for line in f:
        line = line.replace('\n','')
        if line == "< start_bio >":
            continue
        elif line == "< end_bio >":
            data = re.sub(r"[^\w]+", " ", data)
            data = re.sub(r"[\d]+", "<NUM>", data)    
            df_train.loc[len(df_train.index)] = [data , 0]
            data = ""
        else:
            data = data + line + " "

  df_train.to_csv("train_data.csv")

  df_valid = pd.DataFrame(columns=['data', 'label'])

  with open("real.valid.tok",'rt', encoding="utf-8") as f:
    data = ""
    for line in f:
        line = line.replace('\n','')
        if line == "< start_bio >":
            continue
        elif line == "< end_bio >":
            data = re.sub(r"[^\w]+", " ", data)
            data = re.sub(r"[\d]+", "<NUM>", data)    
            df_valid.loc[len(df_valid.index)] = [data , 1]
            data = ""
        else:
            data = data + line + " "

  with open("fake.valid.tok",'rt', encoding="utf-8") as f:
    data = ""
    for line in f:
        line = line.replace('\n','')
        if line == "< start_bio >":
            continue
        elif line == "< end_bio >":
            data = re.sub(r"[^\w]+", " ", data)
            data = re.sub(r"[\d]+", "<NUM>", data)    
            df_valid.loc[len(df_valid.index)] = [data , 0]
            data = ""
        else:
            data = data + line + " "

  df_valid.to_csv("valid_data.csv")
  
  df_test = pd.DataFrame(columns=['data', 'label'])

  with open("real.test.tok",'rt', encoding="utf-8") as f:
    data = ""
    for line in f:
        line = line.replace('\n','')
        if line == "< start_bio >":
            continue
        elif line == "< end_bio >":
            data = re.sub(r"[^\w]+", " ", data)
            data = re.sub(r"[\d]+", "<NUM>", data)    
            df_test.loc[len(df_test.index)] = [data , 1]
            data = ""
        else:
            data = data + line + " "


  with open("fake.test.tok",'rt', encoding="utf-8") as f:
    data = ""
    for line in f:
        line = line.replace('\n','')
        if line == "< start_bio >":
            continue
        elif line == "< end_bio >":
            data = re.sub(r"[^\w]+", " ", data)
            data = re.sub(r"[\d]+", "<NUM>", data)    
            df_test.loc[len(df_test.index)] = [data , 0]
            data = ""
        else:
            data = data + line + " "

  df_test.to_csv("test_data.csv")

def train_LSTM(train_iter, valid_iter, params):
  # LSTM Setting
  vocab_size = params['vocab_size']
  embedding_dim = params['embedding_size']
  hidden_dim = embedding_dim
  output_dim = 1
  model = LSTMnet(vocab_size, embedding_dim, hidden_dim, output_dim)

  optimizer = torch.optim.Adam(model.parameters(), lr = params['lr'])
  criterion = nn.BCELoss()
  # Train LSTM
  model, train_loss_lst, valid_loss_lst, perplexity_list_train, perplexity_list_valid = train(model, train_iter, valid_iter, criterion, optimizer, num_epoch=params['epochs'])
  
  # Save best model
  torch.save(model.state_dict(), "lstm_bestmodel.pt")

def classify_LSTM(test_iter, params):
  vocab_size = params['vocab_size']
  embedding_dim = params['embedding_size']
  hidden_dim = embedding_dim
  output_dim = 1
  model = LSTMnet(vocab_size, embedding_dim, hidden_dim, output_dim)
  model.load_state_dict(torch.load("lstm_bestmodel.pt", map_location=torch.device('cpu')))
  model.eval()
  true_label=torch.LongTensor()
  predicate_label=torch.LongTensor()
  
  for step, batch in enumerate(test_iter):
    text = batch.data[0]
    #label = batch.label.view(-1,1)
    out = model(text)
    out = torch.where(out>0.5,1,0)
    out = out.to("cpu")
    predicate_label = torch.cat((predicate_label,out))
    #true_label = torch.cat((true_label,label))
  
  result_lst = predicate_label.numpy()
  result_lst.shape
  np.count_nonzero(result_lst == 0)
  df_answer = pd.DataFrame(columns=['label'])

  [rows, cols] = result_lst.shape
  #print(rows, cols)
  for i in range(rows):
      for j in range(cols):
          if result_lst[i, j] == 0:
            df_answer.loc[len(df_answer.index)] = ["[FAKE]"]
          else:
            df_answer.loc[len(df_answer.index)] = ["[REAL]"]


  df_answer.to_csv("lstm_blind_result.csv")

def main():
    print(torch.cuda.is_available())
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_hidden', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-embedding_size', type=int, default=256)
    parser.add_argument('-seq_len', type=int, default=300)
    parser.add_argument('-printevery', type=int, default=5000)
    parser.add_argument('-window', type=int, default=3)
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-lr', type=float, default=0.002)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-clip', type=int, default=2.0)
    parser.add_argument('-model', type=str,default='LSTM')
    parser.add_argument('-savename', type=str,default='ffnn')
    parser.add_argument('-loadname', type=str)
    parser.add_argument('-vocab_size', type=int, default=30000)
    parser.add_argument('-sequence_size', type=int, default=300)
    
    params = parser.parse_args()    
    params = vars(params)
    print(params)
    torch.manual_seed(0)

    if params['model'][:4] == 'FFNN':
        # Data Preprocessing for FFNN
        vocab = []
        words = {}
        lookup = {}
        [vocab,words,lookup] = read_encode_custom(["mix.train.tok", "fake.train.tok", "real.train.tok"], 3)
        print("FFNN vocab size:", len(vocab))

        START_TOKEN = words['start_bio'][0]
        END_TOKEN = words['end_bio'][0]
        REAL_TOKEN = words['[REAL]'][0]
        FAKE_TOKEN = words['[FAKE]'][0]
        PAD_TOKEN = words['<pad>'][0]

        mix_train = encode_file(words, "mix.train.tok")
        mix_train_parse, mix_train_labels = parse_mix_corpus(mix_train, START_TOKEN, END_TOKEN, REAL_TOKEN)
        mix_valid = encode_file(words, "mix.valid.tok")
        mix_valid_parse, mix_valid_labels = parse_mix_corpus(mix_valid, START_TOKEN, END_TOKEN, REAL_TOKEN)
        mix_test = encode_file(words, "mix.test.tok")
        mix_test_parse, mix_test_labels = parse_mix_corpus(mix_test, START_TOKEN, END_TOKEN, REAL_TOKEN)
        fake_test = encode_file(words, "fake.test.tok")
        fake_test_parse = parse_corpus(fake_test, START_TOKEN, END_TOKEN)
        fake_test_labels = [False for _ in range(len(fake_test_parse))]
        fake_valid = encode_file(words, "fake.valid.tok")
        fake_valid_parse = parse_corpus(fake_valid, START_TOKEN, END_TOKEN)
        fake_valid_labels = [False for _ in range(len(fake_valid_parse))]
        fake_train = encode_file(words, "fake.train.tok")
        fake_train_parse = parse_corpus(fake_train, START_TOKEN, END_TOKEN)
        fake_train_labels = [False for _ in range(len(fake_train_parse))]
        real_test = encode_file(words, "real.test.tok")
        real_test_parse = parse_corpus(real_test, START_TOKEN, END_TOKEN)
        real_test_labels = [True for _ in range(len(real_test_parse))]
        real_valid = encode_file(words, "real.valid.tok")
        real_valid_parse = parse_corpus(real_valid, START_TOKEN, END_TOKEN)
        real_valid_labels = [True for _ in range(len(real_valid_parse))]
        real_train = encode_file(words, "real.train.tok")
        real_train_parse = parse_corpus(real_train, START_TOKEN, END_TOKEN)
        real_train_labels = [True for _ in range(len(real_train_parse))]
        train_data_ffnn = mix_train_parse + fake_train_parse + real_train_parse
        valid_data_ffnn = mix_valid_parse + fake_valid_parse + real_valid_parse
        test_data_ffnn = mix_test_parse + fake_test_parse + real_test_parse
        train_labels_ffnn = mix_train_labels + fake_train_labels + real_train_labels
        valid_labels_ffnn = mix_valid_labels + fake_valid_labels + real_valid_labels
        test_labels_ffnn = mix_test_labels + fake_test_labels + real_test_labels

        all_label_ffnn = train_labels_ffnn + valid_labels_ffnn + test_labels_ffnn
        all_label_tensor = torch.Tensor(all_label_ffnn)
        all_tensor_ffnn = []
        for s in train_data_ffnn:
            all_tensor_ffnn.append(torch.Tensor(s))
        for s in valid_data_ffnn:
            all_tensor_ffnn.append(torch.Tensor(s))
        for s in test_data_ffnn:
            all_tensor_ffnn.append(torch.Tensor(s))

        all_tensor_ffnn = torch.nn.utils.rnn.pad_sequence(all_tensor_ffnn, True, PAD_TOKEN).type(torch.long)

        train_tensor = all_tensor_ffnn[:len(train_data_ffnn)]
        train_label = all_label_tensor[:len(train_data_ffnn)]
        valid_tensor = all_tensor_ffnn[len(train_data_ffnn):len(train_data_ffnn)+len(valid_data_ffnn)]
        valid_label = all_label_tensor[len(train_data_ffnn):len(train_data_ffnn)+len(valid_data_ffnn)]
        test_tensor = all_tensor_ffnn[len(train_data_ffnn)+len(valid_data_ffnn):]
        test_label = all_label_tensor[len(train_data_ffnn)+len(valid_data_ffnn):]

        print(train_tensor.shape)
        print(valid_tensor.shape)
        print(test_tensor.shape)

        train_dataset = FakeDetectDataset(train_tensor, train_label)
        valid_dataset = FakeDetectDataset(valid_tensor, valid_label)
        test_dataset = FakeDetectDataset(test_tensor, test_label)
        params['pad_size'] = train_tensor.shape[1]

        train_dataset = FakeDetectDataset(train_tensor, train_label)
        valid_dataset = FakeDetectDataset(valid_tensor, valid_label)
        test_dataset = FakeDetectDataset(test_tensor, test_label)
        
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True)
        params['vocab_size'] = len(vocab)

        if params['model'] == 'FFNN_CLASSIFY':
            params['vocab_size'] = len(vocab)
            classify_FFNN(params, START_TOKEN, END_TOKEN, PAD_TOKEN, words)
        else:
            train_FFNN(train_loader, val_loader, test_loader, params)
            test_FFNN(test_loader, params)
        
    if params['model'][:4] == 'LSTM':
        # Data preprocessing for LSTM
        preprocess_data_LSTM()

        # Data Parsing
        train_df = pd.read_csv("train_data.csv")
        valid_df = pd.read_csv("valid_data.csv")
        test_df = pd.read_csv("df_blind.csv")

        print(train_df.shape)
        print(valid_df.shape)
        print(test_df.shape)

        tokenize = lambda x: x.split()
        TEXT = torchdata.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=params['sequence_size'], include_lengths=True, use_vocab=True, batch_first=True)
        LABEL = torchdata.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

        text_data_fields=[
            ("id", None),
            ("data", TEXT),
            ("label", LABEL)
        ]

        train_data, valid_data, test_data = torchdata.TabularDataset.splits(
            path=".", format="csv",
            train="train_data.csv",
            validation="valid_data.csv",
            test="df_blind.csv",
            fields=text_data_fields,
            skip_header=True
        )
        TEXT.build_vocab(train_data, vectors=None, max_size=30000)
        LABEL.build_vocab(train_data)
        params['vocab_size'] = len(TEXT.vocab)
        # vocab_head=TEXT.vocab.freqs.most_common(n=30)
        # vocab_head=pd.DataFrame(data=vocab_head, columns=["word", "frequency"])

        Batch_size= params['batch_size']
        train_iter = torchdata.BucketIterator(train_data, batch_size=Batch_size)
        valid_iter = torchdata.BucketIterator(valid_data, batch_size=Batch_size)
        test_iter = torchdata.BucketIterator(test_data, batch_size=Batch_size)

        if params['model'] == 'LSTM_CLASSIFY':
            classify_LSTM(test_iter, params)
        else:
            train_LSTM(train_iter, valid_iter, params)
            
    

if __name__ == "__main__":
    main()