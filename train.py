from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch import nn
from model import ScopeIt
import numpy as np
import time
import torch
import random
from conlleval import evaluate2
from tqdm import tqdm
import json
import sys

train_filename = sys.argv[1]
seed = int(sys.argv[2])

use_gpu = True
max_length = 128 # max length of a sentence
fine_tune_bert = True # set False to freeze bert
num_layers = 2  # GRU num of layer
hidden_size = 512 # size of GRU hidden layer (in the paper they use 128)
batch_size = 200 # max sentence number in documents
lr = 1e-4 # 1e-4 -> in the paper
dev_ratio = 0.1
tokenizer = None
num_token_labels = 15
repo_path = "/path/to/this/repo"
num_epochs = 30
# pretrained_transformers_model = repo_path + "/../.pytorch_pretrained_bert/bert-base-uncased/"
pretrained_transformers_model = "bert-base-uncased"
# pretrained_transformers_model = "xlm-roberta-base"
# pretrained_transformers_model = "bert-base-multilingual-uncased"
# pretrained_transformers_model = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
model_name = "bert-base"
train_language = "en"
no_token_type = False
only_test = False
dev_set_splitting = "random" # random or any filename
# dev_set_splitting = repo_path + "/data/acl_splits/train/preprocessed/thesis/rq3/dev_set_for_tiny.json" # random or any filename
accumulate_for_n_steps = 16
dev_metric = "only_token"
take_sent_loss = sys.argv[3] == "true"
take_doc_loss = sys.argv[4] == "true"

# model_path = str(max_length) + "_" + str(num_layers) + "_" + str(hidden_size) + "_" + str(batch_size) + "_" + str(lr) + ".pt"
model_path = "test_" + model_name + "_" + train_language + "_acc16.pt"

device_ids = [0, 1, 2, 3, 4, 5, 6] if fine_tune_bert else [0, 1, 2, 3, 4]
# device_ids = [4, 5, 6, 7] if fine_tune_bert else [0, 1, 2, 3, 4]

criterion = torch.nn.BCEWithLogitsLoss()
token_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

label_list = ["B-etime", "B-fname", "B-organizer", "B-participant", "B-place", "B-target", "B-trigger", "I-etime", "I-fname", "I-organizer", "I-participant", "I-place", "I-target", "I-trigger", "O"]
idtolabel = {}
for i,lab in enumerate(label_list):
    idtolabel[i] = lab

if use_gpu and torch.cuda.is_available():
    bert_device = torch.device("cuda:%d"%(device_ids[1]))
    model_device = torch.device("cuda:%d"%(device_ids[0]))
else:
    bert_device = torch.device("cpu")
    model_device = torch.device("cpu")


def read_file(file_name):
    with open(file_name, "r", encoding="utf-8") as fi:
        docs = [json.loads(l) for l in fi]

    return docs

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(texts, all_labels, max_length=max_length):
    t = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
        return_tensors="pt")

    labels = []
    for i, text in enumerate(texts):
        word_ids = t.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-1)
            elif word_idx != previous_word_idx:
                if all_labels:
                    label_ids.append(all_labels[i][word_idx])
                else: # mock label
                    label_ids.append(14) # for "O" label
            else:
                label_ids.append(-1)
            previous_word_idx = word_idx

        labels.append(label_ids)

    labels = torch.LongTensor(labels)
    if no_token_type:
        return labels, (t["input_ids"], t["attention_mask"])
    else:
        return labels, (t["input_ids"], t["attention_mask"], t["token_type_ids"])

def prepare_labels(json_data, max_length=max_length, truncate=False):
    sent_labels = json_data["sent_labels"]
    if truncate:
        sent_labels = sent_labels[:batch_size]

    doc_label = torch.FloatTensor([json_data["doc_label"]])
    sent_labels = torch.FloatTensor(sent_labels)

    return sent_labels, doc_label

def test_model(bert, model, x_test, y_test, sent_avail, token_avail, org_token_labels=[]):
    sent_test_preds = []
    doc_test_preds = []
    all_sent_labels = []
    all_doc_labels = []
    with torch.no_grad():
        test_loss = 0
        all_preds = []
        all_label_ids = []
        for idx, (batch, labels) in enumerate(zip(x_test, y_test)):
            token_labels = batch[0].to(model_device)
            if no_token_type:
                b_input_ids, b_input_mask = tuple(t.to(bert_device) for t in batch[1])
                embeddings = bert(b_input_ids, attention_mask=b_input_mask)[0].detach()
            else:
                b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(bert_device) for t in batch[1])
                embeddings = bert(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)[0].detach()

            embeddings = embeddings.to(model_device)
            sent_labels, doc_label = tuple(t.to(model_device) for t in labels)
            token_out, sent_out, doc_out = model(embeddings)
            sent_out = sent_out.squeeze(0)
            doc_loss = criterion(doc_out.view(-1), doc_label)
            loss = doc_loss

            if sent_avail[idx]:
                sent_loss = criterion(sent_out.view(-1), sent_labels)
                loss += sent_loss
                sent_preds = torch.sigmoid(sent_out).detach().cpu().numpy().flatten()
                sent_labels = sent_labels.detach().cpu().numpy()
                sent_preds = sent_preds[sent_labels != -1].tolist() # some sentences' labels are -1 in acl_split
                sent_labels = sent_labels[sent_labels != -1].tolist() # some sentences' labels are -1 in acl_split
                sent_test_preds += sent_preds
                all_sent_labels += sent_labels

            if token_avail[idx]:
                token_loss = token_criterion(token_out.view(-1, num_token_labels), token_labels.view(-1))
                loss += token_loss
                token_labels = token_labels.detach().cpu().numpy()
                token_preds = token_out.detach().cpu().numpy() # no need for softmax since we only need argmax
                token_preds = np.argmax(token_preds, axis=2) # [Sentences, Seq_len]

                if org_token_labels: # if these are given we can do length fix
                    for i in range(len(org_token_labels[idx])): # For each sentence
                        preds = token_preds[i,:]
                        labs = token_labels[i,:]
                        org_labs = org_token_labels[idx][i]

                        preds = preds[labs != -1].tolist() # get rid of extra subwords, CLS, SEP, PAD
                        preds.extend([14] * (len(org_labs) - len(preds))) # 14 is for "O" label
                        all_preds.extend(preds)
                        all_label_ids.extend(org_labs)
                else:
                    all_preds.extend(token_preds[token_labels != -1].tolist())
                    all_label_ids.extend(token_labels[token_labels != -1].tolist())

            test_loss += loss.item()
            doc_label = doc_label.detach().cpu().numpy()
            all_doc_labels += doc_label.tolist()

            doc_pred = torch.sigmoid(doc_out).detach().cpu().numpy().flatten()
            doc_test_preds += doc_pred.tolist()

    test_loss /= len(x_test)

    doc_test_score = f1_score(all_doc_labels, [ int(x >= 0.5) for x in doc_test_preds], average="macro")

    sent_test_score = 0.0
    if all_sent_labels:
        sent_test_score = f1_score(all_sent_labels, [ int(x >= 0.5) for x in sent_test_preds], average="macro")

    f1 = 0.0
    if all_preds:
        (precision, recall, f1), _ = evaluate2([idtolabel[x] for x in all_label_ids], [idtolabel[x] for x in all_preds])
        precision /= 100
        recall /= 100
        f1 /= 100

    return f1, sent_test_score, doc_test_score, test_loss

def test_whole_set(bert, model, test_set, mode=1, lang="en"):
    x_test = []
    y_test = []
    original_token_labels = []
    sent_avail = []
    token_avail = []
    for d in test_set:
        x_test.append(tokenize_and_align_labels(d["tokens"], d["token_labels"]))
        y_test.append(prepare_labels(d))
        original_token_labels.append(d["token_labels"]) # Necessary for length fix
        sent_avail.append(d["sent"])
        token_avail.append(d["token"])

    token_test_score, sent_test_score, doc_test_score, _ = test_model(bert, model, x_test, y_test, sent_avail, token_avail, org_token_labels=original_token_labels)

    if mode == 1:
        print("%s Document F1 Macro: %.6f."%(lang.upper(), doc_test_score))
    elif mode == 2:
        print("%s Sentence F1 Macro: %.6f."%(lang.upper(), sent_test_score))
    elif mode == 3:
        print("%s Token F1 Macro: %.6f."%(lang.upper(), token_test_score))
    else:
        raise "Mode not defined!"


def build_scopeit(train_data, dev_data, pretrained_model, n_epochs=10, curr_model_path="temp.pt"):
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
    bert = AutoModel.from_pretrained(pretrained_model)

    x_train = []
    y_train = []
    sent_avail = []
    token_avail = []
    for d in train_data:
        if len(d["tokens"]) > batch_size:
            x_train.append(tokenize_and_align_labels(d["tokens"][:batch_size], d["token_labels"]))
            y_train.append(prepare_labels(d, truncate=True))
        else:
            x_train.append(tokenize_and_align_labels(d["tokens"], d["token_labels"]))
            y_train.append(prepare_labels(d, truncate=False))

        sent_avail.append(d["sent"])
        token_avail.append(d["token"])

    x_dev = []
    y_dev = []
    dev_sent_avail = []
    dev_token_avail = []
    # dev_old_token_labels = []
    for d in dev_data:
        x_dev.append(tokenize_and_align_labels(d["tokens"], d["token_labels"]))
        y_dev.append(prepare_labels(d))
        dev_sent_avail.append(d["sent"])
        dev_token_avail.append(d["token"])
        # dev_old_token_labels.append(d["old_token_labels"])

    model = ScopeIt(bert.config.hidden_size, hidden_size, num_layers=num_layers, num_token_labels=num_token_labels)
    # model.load_state_dict(torch.load(repo_path + "/models/scopeit_" + curr_model_path))
    model.to(model_device)

    if torch.cuda.device_count() > 1 and bert_device.type == "cuda":
        bert = nn.DataParallel(bert, device_ids=device_ids[1:])

    # bert.load_state_dict(torch.load(repo_path + "/models/bert_" + curr_model_path))
    bert.to(bert_device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if model_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    total_steps = len(x_train) * n_epochs
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # fine tune bert
    if fine_tune_bert:
        bert_optimizer = torch.optim.AdamW(bert.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(bert_optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

    bert.zero_grad()
    model.zero_grad()
    best_score = -1e6
    best_loss = 1e6

    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = 0
        bert.train()
        model.train()

        # shuffle training data
        train_data = list(zip(x_train, y_train, sent_avail, token_avail))
        random.shuffle(train_data)
        x_train, y_train, sent_avail, token_avail = zip(*train_data)
        ##

        print("Starting Epoch %d"%(epoch+1))
        global_step = 0
        for step, (batch, labels) in enumerate(tqdm(zip(x_train, y_train), desc="Iteration")): # each doc is a batch
            token_labels = batch[0].to(model_device)
            if no_token_type:
                b_input_ids, b_input_mask = tuple(t.to(bert_device) for t in batch[1])
                embeddings = bert(b_input_ids, attention_mask=b_input_mask)[0]
            else:
                b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(bert_device) for t in batch[1])
                embeddings = bert(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)[0]

            if not fine_tune_bert:
                embeddings = embeddings.detach()

            sent_labels, doc_label = tuple(t.to(model_device) for t in labels)
            embeddings = embeddings.to(model_device)
            token_out, sent_out, doc_out = model(embeddings)

            loss = 0.0
            if take_doc_loss:
                doc_loss = criterion(doc_out.view(-1), doc_label)
                loss += doc_loss

            if sent_avail[step] and take_sent_loss:
                sent_out = sent_out.squeeze(0)
                sent_loss = criterion(sent_out.view(-1), sent_labels)
                loss += sent_loss

            if token_avail[step]:
                token_loss = token_criterion(token_out.view(-1, num_token_labels), token_labels.view(-1))
                loss += token_loss

            if loss == 0.0:
                continue

            loss = loss / accumulate_for_n_steps
            loss.backward()
            global_step += 1

            if global_step % accumulate_for_n_steps == 0 or step == len(x_train)-1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # fine tune bert
                if fine_tune_bert:
                    torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
                    bert_optimizer.step()
                    scheduler.step()
                ####

                model_optimizer.step()
                bert.zero_grad()
                model.zero_grad()

            train_loss += loss.item()

        train_loss = train_loss / (len(x_train) / accumulate_for_n_steps)
        elapsed = time.time() - start_time

        # Validation
        bert.eval()
        model.eval()
        token_val_score, sent_val_score, doc_val_score, val_loss = test_model(bert, model, x_dev, y_dev,
                                                                              dev_sent_avail,
                                                                              dev_token_avail)#,
                                                                              # org_token_labels=dev_old_token_labels)
        print("Epoch %d - Train loss: %.4f. Document Validation Score: %.4f. Sentence Validation Score: %.4f. Token Validation Score: %.4f.  Validation loss: %.4f. Elapsed time: %.2fs."% (epoch + 1, train_loss, doc_val_score, sent_val_score, token_val_score, val_loss, elapsed))
        if dev_metric == "only_token":
            val_score = token_val_score
        else:
            val_score = (token_val_score + sent_val_score + doc_val_score) / 3

        if val_score > best_score:
            print("Saving model!")
            torch.save(model.state_dict(), repo_path + "/models/scopeit_" + curr_model_path)
            # bert_to_save = bert.module if hasattr(bert, 'module') else bert  # To handle multi gpu
            torch.save(bert.state_dict(), repo_path + "/models/bert_" + curr_model_path)
            best_score = val_score

        print("========================================================================")

    return bert, model

if __name__ == '__main__':
    if dev_set_splitting == "random":
        train = read_file(train_filename)
        random.shuffle(train)
        dev_split = int(len(train) * dev_ratio)
        dev = train[:dev_split]
        train = train[dev_split:]

    else: # it's a custom filename
        train = read_file(train_filename)
        random.shuffle(train)
        dev = read_file(dev_set_splitting)


    # Test file must contain one more column than others, the "old_token_labels" referring to original token_labels before preprocess_data.py is applied. This is needed in length_fix for testing.
    en_token_test = read_file(repo_path + "/data/acl_splits/test/preprocessed/en_token_test.json")

    print(len([d for d in train if d["token"]]))
    # TODO : print parameters here
    print("max batch size (max sentences in doc): ", batch_size)

    if not only_test:
        bert, model = build_scopeit(train, dev, pretrained_transformers_model, n_epochs=num_epochs, curr_model_path=model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_transformers_model)
        bert = AutoModel.from_pretrained(pretrained_transformers_model)
        model = ScopeIt(bert.config.hidden_size, hidden_size, num_layers=num_layers, num_token_labels=num_token_labels)
        if torch.cuda.device_count() > 1 and bert_device.type == "cuda":
            bert = nn.DataParallel(bert, device_ids=device_ids[1:])

    model.load_state_dict(torch.load(repo_path + "/models/scopeit_" + model_path))
    model.to(model_device)
    bert.load_state_dict(torch.load(repo_path + "/models/bert_" + model_path))
    bert.to(bert_device)
    bert.eval()
    model.eval()

    # EN TOKEN TEST
    test_whole_set(bert, model, en_token_test, mode=3)
