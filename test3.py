from transformers import GPT2LMHeadModel, BertTokenizer
import torch

# 加载预训练的GPT-2中文模型和分词器
model_name = 'uer/gpt2-chinese-cluecorpussmall'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def get_string_probability(prompt, target_string):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    total_prob = 1.0

    for i, char in enumerate(target_string):
        with torch.no_grad():
            outputs = model(input_ids)
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        target_token_id = tokenizer.encode(char, add_special_tokens=False)[0]
        char_prob = next_token_probs[0, target_token_id].item()
        total_prob *= char_prob
        
        # 更新输入，将当前字符添加到输入中
        input_ids = torch.cat([input_ids, torch.tensor([[target_token_id]])], dim=1)
    
    return total_prob

def compare_strings(prompt, target_strings):
    probabilities = {}
    for string in target_strings:
        probabilities[string] = get_string_probability(prompt, string)
    return probabilities

# 示例
prompt = "一个"
target_strings = ["人", "小猫", "大象"]
probabilities = compare_strings(prompt, target_strings)

# 打印结果
for string, prob in probabilities.items():
    print(f"The probability of the next string being '{string}' after '{prompt}' is {prob:.6f}")

# 找出最可能的下一个字符串
most_probable_string = max(probabilities, key=probabilities.get)
print(f"The most probable next string after '{prompt}' is '{most_probable_string}'")
