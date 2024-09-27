
import logging
import pathlib
import torch
# torch.set_num_threads(1)
import transformers

# from GOT.train.trainer import GOTTrainer
# from GOT.train.trainer_vit_llrd import GOTTrainer
from GOT.train.trainer_vit_fixlr import GOTTrainer
from GOT.model import *
from GOT.data import make_supervised_data_module
from GOT.utils.arguments import *
from GOT.utils.constants import *
from GOT.utils.utils import smart_tokenizer_and_embedding_resize
from GOT.model.vision_encoder.vary_b import build_vary_vit_b
import os

# os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['OSS_ENDPOINT'] = "http://oss.i.shaipower.com"

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, padding_side="right", model_max_length=training_args.model_max_length,)


    model = GOTQwenForCausalLM.from_pretrained(model_args.model_name_or_path, use_safetensors=True)



    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token='<|endoftext|>'),
        tokenizer=tokenizer,
        model=model,
        )


    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    vision_tower_dict = model.get_model().initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        freeze_vision_tower=model_args.freeze_vision_tower,
        use_im_start_end=model_args.use_im_start_end,
        vision_select_layer=model_args.vision_select_layer,
        dtype=dtype,
        device=training_args.device
    )

    model.initialize_vision_tokenizer(
        tokenizer=tokenizer, 
        freeze_lm_model=model_args.freeze_lm_model, 
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        device=training_args.device,
    )


    model.to(dtype=dtype, device=training_args.device)
    # 'image_processor_high
    # data_args.image_token_len = vision_tower_dict['image_token_len']
    data_args.image_token_len = 256
    data_args.image_processor = vision_tower_dict['image_processor']
    data_args.image_processor_high = vision_tower_dict['image_processor_high']
    data_args.use_im_start_end = model_args.use_im_start_end

    # mixed relation, to be fixed
    if model_args.freeze_lm_model:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector_vary.parameters():
            p.requires_grad = True
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True


                
    params_grad = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

    

    data_module = make_supervised_data_module(
        interleave=training_args.interleave, 
        with_box=training_args.with_box, 
        tokenizer=tokenizer, 
        data_args=data_args
    )

    trainer = GOTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer._safe_save(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
