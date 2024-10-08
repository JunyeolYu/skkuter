import torch
from transformers.cache_utils import Cache, DynamicCache
import datasets
from transformers.pipelines.pt_utils import KeyDataset
import skkuter_op

class skkuter_pipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = 32000
        
    def convert_batch_to_prompts(self, batches):
        prompts = []
        if isinstance(batches[0], dict):
            batches = [batches]
        for messages in batches:
            prompt = ""
            for message in messages:
                if message['role'] == 'user':
                    prompt += "<|user|> " + message['content'] + "<|end|>"
                elif message['role'] == 'assistant':
                    prompt += "<|assistant|> " + message['content'] + "<|end|>"
            prompt += "<|assistant|> "
            prompts.append(prompt)
        return prompts
    
    def generate(self, prompt, max_new_tokens=50):
        # convert prompts to tensors
        # prompts = self.convert_batch_to_prompts(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        
        batch_size = inputs.input_ids.shape[0]
        seq_len = inputs.input_ids.shape[1]
        
        # create DynamicCache object
        cache = skkuter_op.Cache(batch_size, seq_len, self.model.config.num_hidden_layers)

        # prepare inputs
        model_inputs = self.model.prepare_inputs_for_generation(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            use_cache=True, 
            past_key_values=cache
        )

        # track whether each sequence has finished
        finished_sequences = [False] * batch_size
        # store generated toknes for each sequence
        generated_tokens = [[] for _ in range(batch_size)]

        # Token generation phase
        for _ in range(max_new_tokens):
            res = self.model.forward(**model_inputs)
            
            # extract logits for the last token and select the token
            logits = res['logits']
            next_token_logits = logits[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1)
            
            for i, next_token_id in enumerate(next_token_ids):
                if not finished_sequences[i]:  # if the seq. has not finished
                    generated_tokens[i].append(next_token_id.item())

                    # mark seq. as finished if EOS is generated
                    if next_token_id == self.eos_token_id or next_token_id == 32007:
                        finished_sequences[i] = True

            # break the generation loop
            if all(finished_sequences): break
    
            # prepare new inputs
            model_inputs = {
                'input_ids': next_token_ids.unsqueeze(-1), # newly generated tokens
                'position_ids': model_inputs['position_ids'][:, -1:] + 1,  # update position_ids
                'past_key_values': res['past_key_values'],  # cache update
                'use_cache': True,
                'attention_mask': torch.cat([model_inputs['attention_mask'], torch.ones((batch_size, 1), device=self.model.device)], dim=-1)
            }
            
        # decode tokens using batch_decode
        decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # format the output
        decoded_texts = [[{'generated_text': text}] for text in decoded_texts]
        return decoded_texts
    
    def __call__(
        self,
        prompts,
        batch_size: list = None,
        max_new_tokens=50,
        return_full_text=False,
        temperature=0.0,
        do_sample=False,
        **kwargs
    ):
        if isinstance(prompts, list):
            prompts = self.convert_batch_to_prompts(prompts)
            return self.generate(prompts, max_new_tokens)[0]
        
        elif isinstance(prompts, datasets.Dataset) or isinstance(prompts, KeyDataset):
            if isinstance(prompts, KeyDataset):
                data = prompts.dataset
                key = prompts.key
            else:
                data = prompts
                key = 'message'
            
            # convert chat form to single string
            prompts = self.convert_batch_to_prompts(data[key])

            # run
            idx = 0
            outputs = []
            for bs in batch_size:
                output = self.generate(prompts[idx:idx+bs], max_new_tokens)
                idx+=bs
                outputs += output
            return outputs
        else:
            print("Wrong")