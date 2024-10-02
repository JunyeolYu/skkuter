import torch
from transformers.cache_utils import Cache, DynamicCache
import datasets
from transformers.pipelines.pt_utils import KeyDataset
import skkuter_op
from concurrent.futures import ThreadPoolExecutor

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

    def prepare_batch(self, prompts):
        prompts = self.convert_batch_to_prompts(prompts)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        return inputs

    def generate(self, inputs, max_new_tokens=50):
        # inputs are pre-prepared tensors
        cache = skkuter_op.Cache()
        batch_size = inputs.input_ids.shape[0]
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
            
        return generated_tokens
    
    def decode_generated_tokens(self, generated_tokens):
        # decode tokens using batch_decode
        decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        # format the output
        decoded_texts = [[{'generated_text': text}] for text in decoded_texts]
        return decoded_texts

    def __call__(
        self,
        prompts,
        batch_size=1,
        max_new_tokens=50,
        return_full_text=False,
        temperature=0.0,
        do_sample=False,
        **kwargs
    ):
        if isinstance(prompts, list):
            inputs = self.prepare_batch(prompts)
            generated_tokens = self.generate(inputs, max_new_tokens)
            decoded_texts = self.decode_generated_tokens(generated_tokens)
            return decoded_texts[0]
        elif isinstance(prompts, datasets.Dataset) or isinstance(prompts, KeyDataset):
            if isinstance(prompts, KeyDataset):
                data = prompts.dataset
                key = prompts.key
            else:
                data = prompts
                key = 'message'

            n = len(prompts)
            outputs = []
            executor = ThreadPoolExecutor(max_workers=2)  # Increase max_workers to 2
            i = 0
            # prepare the first batch
            prompt = data[key][i:i + batch_size]
            next_prepare_future = executor.submit(self.prepare_batch, prompt)
            next_decode_future = None
            while i < n:
                # wait for the prepared inputs
                inputs = next_prepare_future.result()
                # start preparing the next batch
                i += batch_size
                if i < n:
                    prompt = data[key][i:i + batch_size]
                    next_prepare_future = executor.submit(self.prepare_batch, prompt)
                else:
                    next_prepare_future = None
                # process the current batch
                generated_tokens = self.generate(inputs, max_new_tokens)
                # start decoding asynchronously
                decode_future = executor.submit(self.decode_generated_tokens, generated_tokens)
                if next_decode_future:
                    # wait for the previous decoding to finish
                    decoded_texts = next_decode_future.result()
                    outputs += decoded_texts
                # set the next decode future
                next_decode_future = decode_future
            # handle the last batch's decoding
            if next_decode_future:
                decoded_texts = next_decode_future.result()
                outputs += decoded_texts
            return outputs
        else:
            print("Wrong")