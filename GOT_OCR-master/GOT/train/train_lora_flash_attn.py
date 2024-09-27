

# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from GOT.utils.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

# from GOT.train.train import train
from GOT.train.train_lora import train

if __name__ == "__main__":
    train()
