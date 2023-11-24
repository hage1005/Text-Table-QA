import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BartTokenizer,BartForConditionalGeneration
import json
from tqdm import tqdm
import random
from pathlib import Path
import logging
import os, sys
from transformers import get_linear_schedule_with_warmup, AdamW
from evaluate_script import get_raw_scores
import argparse
import wandb
from torch.utils.data import  RandomSampler,Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm, trange
from contextlib import nullcontext
import time
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

local_rank = int(os.environ.get("LOCAL_RANK",-1))

def ddp_setup(local_rank):
    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
        

##  添加了AdamW和linear warm up
def create_logger(name, silent=False, to_disk=True, log_file=None):
    """Logger wrapper
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

from openai import OpenAI

client = OpenAI(
    api_key="Your API",
)

def compare_similarity(prompt,answer,model):
    prompt_embeddings = model.encode(prompt,convert_to_tensor = True)
    answers_embeddings = model.encode(answer,convert_to_tensor = True)    
    return F.cosine_similarity(prompt_embeddings,answers_embeddings,dim=0).item()

def get_completion(prompt, model="gpt-4"): 
    prompt+="Only output answer within 20 tokens"
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    #Prevent rate limit
    time.sleep(1)
    return response.choices[0].message.content

class TypeDataset(Dataset):
    def __init__(self, tokenizer, data, is_train, data_use, rerank_link, is_test=0) -> None:
        super(TypeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.ori_data = []
        self.is_train = is_train
        self.is_test = is_test
        self.rerank_link = rerank_link
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        if data_use == 0:    
            self.ori_data = data
        elif data_use == 1:  
            for item in data:
                if sum(item['labels']) != 0:
                    self.ori_data.append(item)
        elif data_use == 2:   
            for item in data:
                if sum(item['labels']) == 1:
                   self.ori_data.append(item)

        total_data = []
        for data in tqdm(self.ori_data):
            path = './Data/HybridQA/WikiTables-WithLinks'
            table_id = data['table_id']
            with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
                table = json.load(f)  
            with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
                requested_document = json.load(f)
            dot_token = self.tokenizer.additional_special_tokens[0]
            dot_token_id = self.tokenizer.convert_tokens_to_ids(dot_token)
            headers = [_[0] for _ in table['header']]
            row_tmp = '{} is {} {}'
            if not is_test:
                answer = data['answer-text']
            type = 0
            if data['type'] == 'comparison':
                type = 1
            if is_train:
                if type == 0:
                    if len(data['row_gold']) > 1:
                        logit = data['row_pre_logit']
                        gold_row = [data['row_gold'][np.argmax(np.array(logit)[data['row_gold']])]]
                    elif len(data['row_gold']) == 0:
                        gold_row = [data['row_pre']]
                    else:
                        gold_row = data['row_gold']
                else:
                    if len(data['row_gold']) == 0:
                        gold_row = np.argsort(-np.array(data['row_pre_logit']))[:3].tolist()
                    elif len(data['row_gold']) > 1:
                        logit = data['row_pre_logit']
                        gold_row = [data['row_gold'][np.argmax(np.array(logit)[data['row_gold']])]]
                        predict_rows = np.argsort(-np.array(data['row_pre_logit']))[:3].tolist()
                        if gold_row[0] in predict_rows:
                            gold_row = predict_rows
                        else:
                            gold_row = [gold_row[0]] + predict_rows[:2]
                    elif len(data['row_gold']) == 1:
                        gold_row = data['row_gold']
                        predict_rows = np.argsort(-np.array(data['row_pre_logit']))[:3].tolist()
                        if gold_row[0] in predict_rows:
                            gold_row = predict_rows
                        else:
                            gold_row = [gold_row[0]] + predict_rows[:2]
            else:
                if type == 0:
                    gold_row = [data['row_pre']]
                else:
                    gold_row = np.argsort(-np.array(data['row_pre_logit']))[:3].tolist()

            if type == 0:
                row = table['data'][gold_row[0]]
                question_ids = self.tokenizer.encode(data['question'])
                prompt_text = data['question']
                input_ids = []
                row_ids = []
                links = []
                for j, cell in enumerate(row):
                    if cell[0] != '':
                        cell_desc = row_tmp.format(headers[j], cell[0], dot_token)
                        prompt_text+=" "+cell_desc
                        cell_toks = self.tokenizer.tokenize(cell_desc)
                        cell_ids = self.tokenizer.convert_tokens_to_ids(cell_toks)
                        row_ids += cell_ids
                    if cell[1] != []:
                        links += cell[1]
                links_ids = []
                if rerank_link:
                    links = self.generate_new_links(links, data)
                
                for idx,link in enumerate(links):
                    if idx == 0:
                        prompt_text += " "+requested_document[link]
                    passage_toks = self.tokenizer.tokenize(requested_document[link])
                    passage_ids = self.tokenizer.convert_tokens_to_ids(passage_toks)
                    links_ids += passage_ids + [dot_token_id]
                    
                generated_answers = get_completion(prompt_text)
                generated_passage_ids = []
                if compare_similarity(prompt_text,generated_answers) >=0.9 and compare_similarity(data['question'],generated_answers) >=0.9:
                    generated_passage_ids = self.tokenizer.encode(generated_answers)

                input_ids = question_ids + row_ids + links_ids+generated_passage_ids+[self.tokenizer.sep_token_id]
            else:
                row_p_dict = data['row_p_dict']
                question_ids = self.tokenizer.encode(data['question'])
                prompt_text = data['question']
                input_ids = question_ids
                for row_id in gold_row:
                    row_ids = []
                    row_links = row_p_dict[row_id]
                    row = table['data'][row_id]
                    links_rank = data['links_rank'][:len(data['links_rank'])//3]
                    if is_train:
                        gold_link = data['gold_link']
                    for j, cell in enumerate(row):
                        if cell[0] != '':
                            cell_desc = row_tmp.format(headers[j], cell[0], dot_token)
                            prompt_text+=" "+cell_desc
                            cell_toks = self.tokenizer.tokenize(cell_desc)
                            cell_ids = self.tokenizer.convert_tokens_to_ids(cell_toks)
                            row_ids += cell_ids
                    if is_train:
                        new_links = list(set(row_links) & set(gold_link) | set(row_links) & set(links_rank))
                    else:
                        new_links = list(set(row_links) & set(links_rank))
                    links = [data['links'][item] for item in new_links]
                    links_ids = []
                    if rerank_link:
                        links = self.generate_new_links(links, data)
                    for idx,link in enumerate(links):
                        if idx == 0:
                            prompt_text += " "+requested_document[link]
                        passage_toks = self.tokenizer.tokenize(requested_document[link])
                        passage_ids = self.tokenizer.convert_tokens_to_ids(passage_toks)
                        links_ids += passage_ids + [dot_token_id]       
                         
                    generated_answers = get_completion(prompt_text)
                    generated_passage_ids = []
                    if compare_similarity(prompt_text,generated_answers) >=0.9 and compare_similarity(data['question'],generated_answers) >=0.9:
                        generated_passage_ids = self.tokenizer.encode(generated_answers)
                    
                    input_ids += row_ids + links_ids + generated_passage_ids + [self.tokenizer.sep_token_id]
            data['input_ids'] = input_ids
            if not is_test:
                answer_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer))
                data['answer_ids'] = [self.tokenizer.eos_token_id] + answer_ids + [self.tokenizer.eos_token_id]
            total_data.append(data)
        self.data = total_data


    def generate_new_links(self, links, data):
        if self.is_train:
            answer_link = [item[2] for item in data['answer-node']]
            new_links, other_links = [], []
            for link in links:
                if link in answer_link:
                    new_links.append(link)
                else:
                    other_links.append(link)
            new_links += other_links
            return new_links
        else:
            links_rank = data['links_rank']
            total_links = data['links']
            row_link_id = [total_links.index(item) for item in links]
            row_link_id_rank = [links_rank.index(item) for item in row_link_id]
            final_rank = np.argsort(row_link_id_rank).tolist()
            new_links = []
            for item in final_rank:
                new_links.append(links[item])
            return new_links

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = self.data[index]
        input_ids = data['input_ids']
        if not self.is_test:
            answer_ids = data['answer_ids']
            return input_ids, answer_ids, data
        else:
            return input_ids, [0], data
            
                    
def collate(data, tokenizer, max_length, is_test=0):
    bs = len(data)
    max_input_length = 0
    input_ids = []
    answer_ids = []
    metadata = []
    max_input_length = max([len(item[0]) for item in data])
    max_answer_length = max([len(item[1]) for item in data])
    if max_input_length > max_length:
        max_input_length = max_length
    for i in range(bs):
        if len(data[i][0]) > max_input_length:
            input_id = data[i][0][:max_input_length]
        else:
            input_id = data[i][0] + (max_input_length - len(data[i][0])) * [tokenizer.pad_token_id]
        input_ids.append(input_id)

        if not is_test:
            if len(data[i][1]) > max_answer_length:
                answer_id = data[i][1][:max_answer_length]
            else:
                answer_id = data[i][1] + (max_answer_length - len(data[i][1])) * [tokenizer.pad_token_id]
            answer_ids.append(answer_id)
        metadata.append(data[i][2])
    input_ids = torch.tensor(input_ids)
    input_mask = torch.where(input_ids==tokenizer.pad_token_id, 0, 1)
    if not is_test:
        answer_ids = torch.tensor(answer_ids)
        answer_mask = torch.where(answer_ids==tokenizer.pad_token_id, 0, 1)
        return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "answer_ids":answer_ids.cuda(), "answer_mask":answer_mask.cuda(), "metadata":metadata}
    else:
        return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "metadata":metadata}


def eval(args,tokenizer, model, loader):
    model.eval()
    total, acc = 0, 0
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader,desc="Evaluate")):
            if args.local_rank != -1:
                generated_ids = model.module.generate(
                    input_ids = data['input_ids'],
                    attention_mask=data['input_mask'],
                    max_length = 20,
                    num_beams = 3,
                    early_stopping = True
                )
            else:
                generated_ids = model.generate(
                    input_ids = data['input_ids'],
                    attention_mask=data['input_mask'],
                    max_length = 20,
                    num_beams = 3,
                    early_stopping = True
                )                
            metadata = data['metadata']
            for i, item in enumerate(generated_ids):
                total += 1
                output = torch.masked_select(item, item.ge(3))
                pred_answer = tokenizer.decode(output)
                answer = metadata[i]['answer-text']
                question_id = metadata[i]['question_id']

                if pred_answer == answer:
                    acc += 1
                outputs.append({"question_id":question_id, "pred":pred_answer})
    print(f"outputs num: {len(outputs)}")
    return acc / total, outputs

def test_eval(args, tokenizer, model, loader):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            generated_ids = model.generate(
                input_ids = data['input_ids'],
                attention_mask=data['input_mask'],
                max_length = 20,
                num_beams = 3,
                early_stopping=True
            )
            metadata = data['metadata']
            for i, item in enumerate(generated_ids):
                output = torch.masked_select(item, item.ge(3))
                pred_answer = tokenizer.decode(output)
                question_id = metadata[i]['question_id']
                outputs.append({"question_id":question_id, "pred":pred_answer})
    print(f"outputs num: {len(outputs)}")
    return outputs


def main():
    device = torch.device("cuda")
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptm_type', type=str, default='facebook/bart-large', help='Pre-trained model to use')
    parser.add_argument('--train_data_path', type=str, default='./Data/HybridQA/train.row.json', help='Path to train data')
    parser.add_argument('--dev_data_path', type=str, default='./Data/HybridQA/test.row.json', help='Path to dev data')
    parser.add_argument('--predict_save_path', type=str, default='./Data/HybridQA/test_answers.json', help='Path to save predictions')
    
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epoch_nums', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate for training')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay for Adam optimizer')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient Accumulation for Adam optimizer')
    parser.add_argument('--warmup_steps', type=int, default=0.1, help='Number of warmup steps for learning rate scheduler')
    
    parser.add_argument('--is_train', type=int, default=1, help='Whether to train the model')
    parser.add_argument('--is_test', type=int, default=0, help='Whether to test the model')
    parser.add_argument('--seed', type=int, default=2001, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='read2', help='Output directory for saving model and logs')
    parser.add_argument('--load_dir', type=str, default='read1', help='Directory for loading model')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum length of input sequence')

    args = parser.parse_args()

    ptm_type = args.ptm_type
    train_data_path = args.train_data_path
    dev_data_path = args.dev_data_path
    predict_save_path = args.predict_save_path
    batch_size = args.batch_size
    epoch_nums = args.epoch_nums
    learning_rate = args.learning_rate
    adam_epsilon = args.adam_epsilon
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    
    is_train = args.is_train
    is_test = args.is_test
    seed = args.seed
    output_dir = args.output_dir
    load_dir = args.load_dir
    max_length = args.max_length

    
    args.local_rank = local_rank
    set_seed(seed)

    log_file = 'log.txt'
    ckpt_file = 'ckpt_best.pt'
    load_ckpt_file = 'ckpt.pt'
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        ddp_setup(local_rank)
        device = torch.device("cuda",local_rank)
    print(f"Process rank: {local_rank}, device: {device}, distributed training: {bool(local_rank != -1)}")

    with open('Data/HybridQA/dev_reference.json', 'r') as f:
        reference = json.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(load_dir).mkdir(parents=True, exist_ok=True)
    logger = create_logger("Training", log_file=os.path.join(output_dir, log_file))

    train_data, dev_data = load_data(train_data_path), load_data(dev_data_path)
    tokenizer = BartTokenizer.from_pretrained(ptm_type)     
    model = BartForConditionalGeneration.from_pretrained(ptm_type).to(device)
    special_tokens_dict = {'additional_special_tokens': ['<dot>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    model_load_path = os.path.join(load_dir, load_ckpt_file)
    logger.info(f"loading trained parameters from {model_load_path}")
    model.load_state_dict(torch.load(model_load_path))
    
    if local_rank in [-1,0]:
        wandb.init(project="711 HW4",name="Bart-large")
        wandb.config.update(args) 
    
    if is_train:
        if os.path.exists(os.path.join(load_dir,"train_dataset.pt")):
            train_dataset = torch.load(os.path.join(load_dir,"train_dataset.pt"))
        else:
            train_dataset = TypeDataset(tokenizer, train_data, is_train=1, data_use=0, rerank_link=1)
            torch.save(train_dataset, os.path.join(load_dir,"train_dataset.pt"))
        train_sampler = RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset) 
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=lambda x: collate(x, tokenizer, max_length))
       
    if is_test: 
        if os.path.exists(os.path.join(load_dir,"test_dataset.pt")):
            dev_dataset = torch.load(os.path.join(load_dir,"test_dataset.pt"))
        else:
            dev_dataset = TypeDataset(tokenizer, dev_data, is_train=0, data_use=0, rerank_link=1, is_test=is_test)
            torch.save(dev_dataset, os.path.join(load_dir,"test_dataset.pt"))
    else:
        if os.path.exists(os.path.join(load_dir,"dev_dataset.pt")):
            dev_dataset = torch.load(os.path.join(load_dir,"dev_dataset.pt"))
        else:
            dev_dataset = TypeDataset(tokenizer, dev_data, is_train=0, data_use=0, rerank_link=1, is_test=is_test)
            torch.save(dev_dataset, os.path.join(load_dir,"dev_dataset.pt")
                       )        
    dev_loader = DataLoader(dev_dataset,shuffle=False, batch_size=batch_size, collate_fn=lambda x: collate(x, tokenizer, max_length, is_test))
    if args.local_rank != -1:
        torch.distributed.barrier()


    if is_train:
        best_acc = -1                
        if local_rank != -1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)    
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate, eps=adam_epsilon)
        
        t_total = len(train_dataset) // (batch_size*args.gradient_accumulation_steps) * epoch_nums
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(warmup_steps*t_total), num_training_steps=t_total
        )

        model.zero_grad()
        train_iterator = trange(int(epoch_nums), desc="Epoch")
        global_step = 0
        tr_loss = 0.0

        for epoch in train_iterator:
            if local_rank!=-1:
                train_loader.sampler.set_epoch(epoch)
            epoch_iterator = tqdm(train_loader, desc="Training Bart continuously ",disable=args.local_rank not in [-1,0])
            for step, data in enumerate(epoch_iterator):
                model.train()
                labels = data['answer_ids'][:, 1:].clone()
                labels[labels==tokenizer.pad_token_id] = -100
                mcontext = model.no_sync if local_rank !=-1 and (step+1) % args.gradient_accumulation_steps != 0 else nullcontext    
                with mcontext():            
                    loss = model(input_ids=data['input_ids'], 
                                    attention_mask=data['input_mask'], 
                                    decoder_input_ids=data['answer_ids'][:, :-1],
                                    labels=labels).loss
                    loss = loss/args.gradient_accumulation_steps
                    loss.backward()
                tr_loss += loss.item()
                if (step+1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    global_step += 1
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    if args.local_rank in [-1,0]:
                        epoch_iterator.set_description(f"Loss:{tr_loss/global_step}")
                        wandb.log(
                            {   
                                'loss': tr_loss/global_step,
                                'learning_rate': optimizer.param_groups[0]['lr']
                            }
                        )
            if local_rank in [-1,0]:
                print(f"Rank {args.local_rank} Start validation at epoch {epoch}")
                total_exact, outputs = eval(args,tokenizer, model, dev_loader)
        
                scores = get_raw_scores(outputs, reference)
                total_exact = scores['total exact']
                for key, value in scores.items():
                    logger.info(f"{key:25}: {value}")
                wandb.log(scores)
                
                if total_exact > best_acc:
                    best_acc = total_exact
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_save_path = os.path.join(output_dir, ckpt_file)
                    logger.info(f"saving model...to {model_save_path}")
                    torch.save(model_to_save.state_dict(), model_save_path)
                    # model_to_save.save_pretrained(model_save_path)
                if args.local_rank ==0:
                    torch.distributed.barrier()
            else:
                torch.distributed.barrier()
    else:
        model_load_path = os.path.join(output_dir, ckpt_file)
        logger.info(f"loading trained parameters from {model_load_path}")
        # model = PeftModel.from_pretrained(model, model_load_path).to(device)
        model.load_state_dict(torch.load(model_load_path))
        logger.info("start eval....")
        if not is_test:
            _, outputs = eval(args,tokenizer, model, dev_loader)
            scores = get_raw_scores(outputs, reference)
            total_exact = scores['total exact']
            for key, value in scores.items():
                logger.info(f"{key:25}: {value}")
            with open(predict_save_path, 'w') as f:
                json.dump(outputs, f, indent=2)
            logger.info(f"answer save to {predict_save_path}")
        else:
            outputs = test_eval(args,tokenizer, model, dev_loader)          
            with open(predict_save_path, 'w') as f:
                json.dump(outputs, f, indent=2)
            logger.info(f"answer save to {predict_save_path}")
    
    if args.local_rank != -1:
        destroy_process_group()
        
if __name__ == '__main__':
    main()