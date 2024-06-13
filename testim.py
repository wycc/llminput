import itertools
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_cin_file(typ, filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    key_to_character = {}
    if type == 'raw':
        is_def = True
    else:
      is_def = False
    endkey=r'[ ]' # This is default endkey, which is used when %endkey is not found in the file
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):  # 忽略空行和注释行
            continue
        # handle %chardef to load key sequence only in the chardef section
        if typ == 'cin':
          if line.startswith("%chardef"):
              tokens = line.split()
              if tokens[1] == 'begin':
                  is_def = True
              elif tokens[1] == 'end':
                  is_def = False
              continue
          elif line.startswith("%endkey"):
              tokens = line.split()
              endkey = '['+tokens[1]+' ]'
              continue
          if is_def == False:
              continue
        # split the line by any characters in endkey
        parts = line.split(' ')

        if len(parts) >= 2:
            if typ == 'raw':
              key_sequence = parts[0].upper()
              character = parts[1]
            else:
              key_sequence = parts[1].upper()
              character = parts[0]
                
            if key_sequence in key_to_character:
                key_to_character[key_sequence].append(character)
            else:
                key_to_character[key_sequence] = [character]
    
    return key_to_character

def convert_key_sequence_to_text(key_sequence, key_to_character):
    key_sequence = key_sequence.upper()
    #split the key sequence by space
    key_sequence = key_sequence.split()
    # Each key sequence might map to multiple words, we need to generate all posible combinations
    # of words that can be generated from the key sequence
    # We can use itertools.product to generate all possible combinations of words
    # from the key sequence. We can then join the words together to form the final text.
    words = [key_to_character[key] for key in key_sequence]
    all_combinations = itertools.product(*words)
    all=[]
    for combination in all_combinations:
        all.append(''.join(combination))
    return all

# 示例Cin格式文件路径
#cin_file_path = '14_dayi.cin'
cin_file_path = 'lookup-bpmf.cin'

# 载入Cin格式输入法表格
#key_to_character = load_cin_file('raw',cin_file_path)
key_to_character = load_cin_file('cin',cin_file_path)

# 示例按键序列
key_sequence = 'u ek7 bp6 2k7 g6'
# 转换按键序列为文字
text = convert_key_sequence_to_text(key_sequence, key_to_character)
print(f"转换后的文字: {text}")

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math

# 加載 GPT-2 模型和 tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
# 將模型設置為評估模式
model.eval()

def calculate_probabilities(texts):
    # 將多個文本編碼為 tokens
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # 計算 tokens 的 log 概率
    with torch.no_grad():
        input_ids=input_ids.to(device)
        outputs = model(input_ids, attention_mask=attention_mask.to(device), labels=input_ids)

    # 轉換 logits 為概率
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1).to('cpu')


    # 計算每個字串的總 log 概率
    batch_size, seq_len = input_ids.size()
    total_log_probs = []

    for i in range(batch_size):
        seq_log_prob = 0.0
        for j in range(seq_len):
            if attention_mask[i, j] == 1:  # 只考慮非填充部分
                token_id = input_ids[i, j].item()
                token_log_prob = log_probs[i, j, token_id].item()
                seq_log_prob += token_log_prob
        total_log_probs.append(seq_log_prob)

    # 計算每個文本的總概率
    probabilities = [math.exp(log_prob) for log_prob in total_log_probs]

    return probabilities

# segment text into batches of 64 to avoid OOM error
text_batches = [text[i:i+64] for i in range(0, len(text), 64)]

# calculate probabilities for each batch
probs_batches = []
for batch in text_batches:
    probs_batch = calculate_probabilities(batch)
    probs_batches.append(probs_batch)

# concatenate probabilities for all batches
probs = [prob for probs_batch in probs_batches for prob in probs_batch]


# sort texts by probability in descending order
sorted_probs, sorted_texts = zip(*sorted(zip(probs, text), reverse=True))

for k, v in zip(sorted_probs, sorted_texts):
    print(f"{k:.20g} {v}")


