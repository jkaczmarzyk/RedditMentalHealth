import torch 
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from sklearn import metrics

class CustomDataSet(Dataset):

    def __init__(self,dataframe,tokenizer,max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)
    
    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids,dtype=torch.long),
            'mask': torch.tensor(mask,dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class BERTClass(torch.nn.Module):

    def __init__(self,model_name):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(model_name,hidden_size)
        self.l2 = torch.nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(hidden_size,1)
    
    def forward(self,ids,mask, token_type_ids):
        _ , output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)

        return output

def loss_fn(outputs,targets):
    return torch.nn.BCEWithLogitsLoss()(outputs,targets)

def validation(epoch,hyperparameters,device,training_loader,model):
    model.eval()
    fin_targets = []
    fin_output = []

    with torch.no_grad():
        for _,data in enumerate(training_loader,0):
            ids = data['ids'].to(device,dtype=torch.long)
            mask = data['mask'].to(device,dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device,dtype=torch.long)
            targets = data['targets'].to(device,dtype=torch.float)
            s = targets.size(dim=0)
            targets = targets.resize(s,1)
            outputs = model(ids,mask,token_type_ids)

            t = targets.cpu().detach().numpy()
            for item in t:
                fin_targets.append(item[0])
            
            o = (torch.sigmoid(outputs) > 0.5).float().cpu().detach().numpy()
            for item in outputs:
                fin_output.append(torch.sigmoid(item).float().cpu().detach().numpy()[0])

            loss = loss_fn(outputs,targets)
            accuracy = metrics.accuracy_score(t,o)
            recall_micro = metrics.recall_score(t,o,average='micro')
            recall_macro = metrics.recall_score(t,o,average='macro')
            precision_micro = metrics.precision_score(t,o,average='micro')
            precision_macro = metrics.precision_score(t,o,average='macro')
            f1_score_micro = metrics.f1_score(t,o,average='micro')
            f1_score_macro = metrics.f1_score(t,o,average='macro')

    return accuracy, recall_macro, recall_micro, precision_macro, precision_micro, f1_score_macro, f1_score_micro, loss

def train(epoch,hyperparameters,device,training_loader,validation_loader,model,optimizer):

    param_dic = {'train':{},'validation':{}}

    accs = []
    recalls_mac = []
    recalls_mic = []
    precisions_mac = []
    precisions_mic = []
    f1s_mic = []
    f1s_mac = []
    losses = []

    v_accs = []
    v_recalls_mac = []
    v_recalls_mic = []
    v_precisions_mac = []
    v_precisions_mic = []
    v_f1s_mic = []
    v_f1s_mac = []
    v_losses = []

    model.train()
    example_ct = 0
    chk_count = 0
    for _,data in enumerate(training_loader,0):
        ids = data['ids'].to(device,dtype=torch.long)
        mask = data['mask'].to(device,dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device,dtype=torch.long)
        targets = data['targets'].to(device,dtype=torch.float)
        s = targets.size(dim=0)
        targets = targets.resize(s,1)

        outputs = model(ids,mask,token_type_ids)

        #maybe get rid of this! (zero_grad()) bad?
        optimizer.zero_grad()

        loss = loss_fn(outputs,targets)

        step_loss = loss.item()

        t = targets.cpu().detach().numpy()

        # makes outputs binary 
        o = (torch.sigmoid(outputs) > 0.5).float().cpu().detach().numpy()

        # calculating metrics for training
        if _ % hyperparameters['log_freq'] == 0:
            accuracy = metrics.accuracy_score(t,o)
            recall_micro = metrics.recall_score(t,o,average='micro')
            recall_macro = metrics.recall_score(t,o,average='macro')
            precision_micro = metrics.precision_score(t,o,average='micro')
            precision_macro = metrics.precision_score(t,o,average='macro')
            f1_score_micro = metrics.f1_score(t,o,average='micro')
            f1_score_macro = metrics.f1_score(t,o,average='macro')

            accs.append(accuracy)
            recalls_mac.append(recall_macro)
            recalls_mic.append(recall_micro)
            precisions_mac.append(precision_macro)
            precisions_mic.append(precision_micro)
            f1s_mic.append(f1_score_micro)
            f1s_mac.append(f1_score_macro)
            losses.append(loss)

            val_metrics = validation(epoch,hyperparameters,device,validation_loader,model)
            v_accs.append(val_metrics[0])
            v_recalls_mac.append(val_metrics[1])
            v_recalls_mic.append(val_metrics[2])
            v_precisions_mac.append(val_metrics[3])
            v_precisions_mic.append(val_metrics[4])
            v_f1s_mac.append(val_metrics[5]) 
            v_f1s_mic.append(val_metrics[6])
            v_losses.append(val_metrics[7])

            print('Checkpoint {} | F1 : {} | Rec : {} | Prec : {} |'.format(
                chk_count,f1_score_macro,recall_macro,precision_macro)
                )
            chk_count += 1

        #maybe get rid of this!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # put training parameters into a dictionary
    param_dic['train'] = {
        'accuracy': accs,
        'recall_macro': recalls_mac,
        'recall_micro': recalls_mic,
        'precisions_mac': precisions_mac,
        'precisions_mic': precisions_mic,
        'f1s_mac': f1s_mac,
        'f1s_mic': f1s_mic,
        'losses': losses
        }
    
    # put validaton parameters into a dictionary
    param_dic['validation'] = {
        'accuracy': v_accs,
        'recall_macro': v_recalls_mac,
        'recall_micro': v_recalls_mic,
        'precisions_mac': v_precisions_mac,
        'precisions_mic': v_precisions_mic,
        'f1s_mac': v_f1s_mac,
        'f1s_mic': v_f1s_mic,
        'losses': v_losses
        }

    return param_dic
