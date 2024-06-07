import itertools

def load_cin_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    key_to_character = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):  # 忽略空行和注释行
            continue
        parts = line.split()
        if len(parts) >= 2:
            key_sequence = parts[0]
            character = parts[1]
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
cin_file_path = '14_dayi.cin'

# 载入Cin格式输入法表格
key_to_character = load_cin_file(cin_file_path)

# 示例按键序列
key_sequence = 'E A7SO A'
# 转换按键序列为文字
text = convert_key_sequence_to_text(key_sequence, key_to_character)
print(f"转换后的文字: {text}")
