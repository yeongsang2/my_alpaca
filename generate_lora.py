import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import fire


def main(
    base_model: str = "",
    lora_weights: str = ""
    ):

    model = AutoModelForCausalLM.from_pretrained(base_model, load_in_8bit=True, device_map={"":0})
    model = PeftModel.from_pretrained(model, lora_weights)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model.eval()

    def gen(x): 
        q = f"###명렁어: {x}\n\n### 응답:"
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