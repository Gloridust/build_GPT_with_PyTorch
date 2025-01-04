import torch
from model import GPTModel
from tokenizer import Tokenizer

def generate(model, tokenizer, text, max_length, device):
    input, att_mask = tokenizer.encode(text)
    input = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(0)
    
    stop = False
    input_len = len(input[0])
    
    while not stop:
        if len(input[0]) - input_len > max_length:
            next_symbol = tokenizer.sep_token
            input = torch.cat(
                [input.detach(), torch.tensor([[next_symbol]], dtype=input.dtype, device=device)], -1)
            break
            
        projected, _ = model(input)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        
        if next_symbol == tokenizer.sep_token:
            stop = True
            
        input = torch.cat(
            [input.detach(), torch.tensor([[next_symbol]], dtype=input.dtype, device=device)], -1)
    
    decode = tokenizer.decode(input[0].tolist())
    decode = decode[len(text):]
    return "".join(decode)

def get_device():
    """获取可用的设备,优先级: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main():
    model_path = "output/best.pt"
    vocab_path = "data/vocab.json"
    max_length = 128
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = Tokenizer(vocab_path)
    
    model_param = {
        "d_model": 768,
        "d_ff": 2048,
        "d_k": 64,
        "d_v": 64,
        "n_layers": 6,
        "n_heads": 8,
        "max_pos": 1800,
        "device": device,
        "vocab_size": tokenizer.get_vocab_size(),
    }
    
    model = GPTModel(**model_param)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("模型加载完成,开始对话...")
    while True:
        text = input("\n用户: ")
        if not text:
            continue
        if text.lower() in ["q", "quit", "exit"]:
            print("对话结束")
            break
            
        response = generate(model, tokenizer, text, max_length, device)
        print(f"AI: {response}")

if __name__ == '__main__':
    main() 