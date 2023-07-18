import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import fire


def main(
    base_model: str = "beomi/KoAlpaca-Polyglot-12.8B",
    lora_weights: str = "/content/my_alpaca/output/"
    ):
    config = PeftConfig.from_pretrained("/content/my_alpaca/output/checkpoint-100/")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
    model = PeftModel.from_pretrained(model, "/content/my_alpaca/output/checkpoint-100/")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    model.eval()

    def gen(x):
        q = f"###instruction: {x}\n\n### output:"
        # print(q)
        gened = model.generate(
            **tokenizer(
                q, 
                return_tensors='pt', 
                return_token_type_ids=False
            ).to('cuda'), 
            max_new_tokens=100,
            early_stopping=True,
            do_sample=True,
            eos_token_id=2,
        )
        print(tokenizer.decode(gened[0]))
    
    while(True):
        user_input = input("Enter your input (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        gen(user_input)

if __name__ == "__main__":
    fire.Fire(main)