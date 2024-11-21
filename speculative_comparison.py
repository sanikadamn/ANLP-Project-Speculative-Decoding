import os
if not os.path.exists("/scratch/sarthak"):
    os.makedirs("/scratch/sarthak")

os.environ["HF_HOME"] = "/scratch/sarthak/"

import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple, Optional
import time
import numpy as np

from typing import Tuple
import torch
import datasets
from transformers import T5Tokenizer

en_gr_dataset = datasets.load_dataset('wmt16', 'de-en')

class SpeculativeDecoder:
    def __init__(
        self,
        target_model_name = "google-t5/t5-3b",
        draft_model_name = "google-t5/t5-small",
        device = "cuda" if torch.cuda.is_available() else "cpu",
        gamma = 4,
        temperature = 0.5
    ):
        self.device = device
        self.gamma = gamma

        self.temperature = temperature

        self.tokenizer = T5Tokenizer.from_pretrained(target_model_name)
        
        # self.target_model = T5ForConditionalGeneration.from_pretrained(target_model_name).to(device)
        self.target_model = AutoModelForSeq2SeqLM.from_pretrained(target_model_name, device_map='auto')

        self.draft_model = T5ForConditionalGeneration.from_pretrained(draft_model_name).to(device)

        self.target_model.eval()
        self.draft_model.eval()

    def get_draft_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        gamma: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get draft logits for gamma tokens"""
        draft_tokens = []
        draft_probs = []
        current_decoder_ids = decoder_input_ids

        # Generate gamma tokens from the draft model
        for _ in range(gamma):
            with torch.no_grad():
                outputs = self.draft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=current_decoder_ids,
                    return_dict=True
                )
                logits = outputs.logits[:, -1, :]  # Get logits for last position
                probs = F.softmax(logits, dim=-1)

                # Sample token
                token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                prob = probs.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)

                draft_tokens.append(token_id.item())
                draft_probs.append(prob.item())

                # Update decoder inputs for next iteration
                current_decoder_ids = torch.cat(
                    [current_decoder_ids, token_id.view(1, 1)],
                    dim=1
                )

                if token_id.item() == self.tokenizer.eos_token_id:
                    break

        return draft_tokens, draft_probs#, current_decoder_ids, outputs.logits

    # def get_target_probs(
    #     self,
    #     input_ids: torch.Tensor,
    #     attention_mask: torch.Tensor,
    #     decoder_input_ids: torch.Tensor,
    #     draft_tokens: torch.Tensor
    # ) -> torch.Tensor:
    #     """Get target probabilities for the draft tokens in parallel."""
    #     with torch.no_grad():
    #         # Add draft tokens to decoder input
    #         # full_decoder_ids = torch.cat([decoder_input_ids, draft_tokens.unsqueeze(0)], dim=1)

    #         full_decoder_ids = [decoder_input_ids]
    #         for i in range(len(draft_tokens)):
    #             x = torch.cat([decoder_input_ids, draft_tokens.unsqueeze(0)[:, :i+1]], dim=1)
    #             full_decoder_ids.append(x)

    #         maxlen = max([x.shape[1] for x in full_decoder_ids])

    #         padded_decoder_ids = torch.stack([torch.tensor(
    #             F.pad(x, (0, maxlen - x.shape[1]), value=self.tokenizer.pad_token_id)[0]
    #         , device=self.device) for x in full_decoder_ids])

    #         batch_size = padded_decoder_ids.shape[0]
    #         input_ids_batched = input_ids.repeat(batch_size, 1)
    #         attention_mask_batched = attention_mask.repeat(batch_size, 1)

    #         # make it a triangular attention mask


    #         # print(input_ids_batched.shape, attention_mask_batched.shape, padded_decoder_ids.shape)

    #         # outputs = self.target_model(
    #         #     input_ids=input_ids,
    #         #     attention_mask=attention_mask,
    #         #     decoder_input_ids=full_decoder_ids,
    #         #     return_dict=True
    #         # )
    #         outputs = self.target_model(
    #             input_ids=input_ids_batched,
    #             attention_mask=attention_mask_batched,
    #             decoder_input_ids=padded_decoder_ids,
    #             # decoder_attention_mask=torch.triu(
    #             #     torch.zeros((padded_decoder_ids.shape[0], padded_decoder_ids.shape[1]), device=self.device)
    #             # ),
    #             return_dict=True
    #         )
    #         # print("passes target model")

    #         batched_logits = outputs.logits
    #         # print(batched_logits.shape)
    #         # 5, 5, 32128
    #         logits = torch.zeros((batched_logits.shape[1], batched_logits.shape[2]), device=self.device)
    #         for i in range(len(draft_tokens)):
    #             logits[i] = batched_logits[i, i, :]

    #         # print(logits.shape, batched_logits[-1].shape)

    #         target_probs = F.softmax(logits, dim=-1)

    #         # # Get probabilities for positions before each draft token
    #         # logits = outputs.logits[:, -(len(draft_tokens) + 1):-1, :]
    #         # target_probs = F.softmax(logits, dim=-1)

    #         # print(batched_logits[-1].unsqueeze(0).shape)

    #         return target_probs.squeeze(0), logits.unsqueeze(0)

    def get_target_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        draft_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Get target probabilities for the draft tokens in parallel."""
        with torch.no_grad():
            # Add draft tokens to decoder input
            full_decoder_ids = torch.cat([decoder_input_ids, draft_tokens.unsqueeze(0)], dim=1)

            # print(input_ids_batched.shape, attention_mask_batched.shape, padded_decoder_ids.shape)

            outputs = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=full_decoder_ids,
                # decoder_attention_mask=torch.triu(
                #     torch.zeros((full_decoder_ids.shape[0], full_decoder_ids.shape[1]), device=self.device)
                # ),
                return_dict=True
            )

            # Get probabilities for positions before each draft token
            logits = outputs.logits[:, -(len(draft_tokens) + 1):-1, :]
            target_probs = F.softmax(logits, dim=-1)

            return target_probs.squeeze(0), outputs.logits

    # def verify_tokens(
    #     self,
    #     target_probs: torch.Tensor,
    #     draft_tokens: torch.Tensor,
    #     draft_probs: torch.Tensor,
    # ) -> int:
    #     """Determine number of accepted tokens"""
    #     # Get target probabilities for the draft tokens
    #     target_probs_draft_tokens = target_probs.gather(
    #         -1,
    #         draft_tokens.unsqueeze(-1)
    #     ).squeeze(-1)

    #     # Calculate acceptance ratios
    #     acceptance_ratios = target_probs_draft_tokens / draft_probs

    #     # Sample uniform random numbers
    #     random_nums = torch.rand_like(acceptance_ratios)

    #     # Find number of accepted tokens
    #     # Accept if random number < min(1, target_prob / draft_prob)
    #     accepts = random_nums < torch.minimum(
    #         torch.ones_like(acceptance_ratios),
    #         acceptance_ratios
    #     )

    #     # Find first rejection
    #     try:
    #         n_accepted = torch.where(~accepts)[0][0].item()
    #     except:
    #         n_accepted = len(accepts)

    #     return n_accepted

    def verify_tokens(
        self,
        target_probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        temperature: float = 1.0
    ) -> int:
        """Determine number of accepted tokens"""
        # Get target probabilities for the draft tokens
        target_probs_draft_tokens = target_probs.gather(
            -1,
            draft_tokens.unsqueeze(-1)
        ).squeeze(-1)

        target_probs_draft_tokens = target_probs_draft_tokens / max(temperature, 1e-10)
        draft_probs = draft_probs / max(temperature, 1e-10)

        # Calculate acceptance ratios
        acceptance_ratios = target_probs_draft_tokens.float() / draft_probs

        # Sample uniform random numbers 
        random_nums = torch.zeros_like(target_probs_draft_tokens).float().uniform_()

        mask = random_nums > acceptance_ratios
        num_accepted = (mask.cumsum(dim = -1) == 0).sum(dim = -1)

        return num_accepted.int().item()

    def translate(
        self,
        source_text: str,
        max_length: int = 128
    ) -> str:
        """Translate source text using speculative decoding."""
        # Encode source text
        encoder_inputs = self.tokenizer(
            f"translate English to German: {source_text}",
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Initialize with start token
        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]], device=self.device)

        output = self.target_model(
            input_ids=encoder_inputs.input_ids,
            attention_mask=encoder_inputs.attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

        probs = output.logits[:, -1, :]
                    
        probs = F.softmax(probs, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1)

        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id, token_id.item()]], device=self.device)


        total_tokens = 0
        accepted_tokens = 0

        while decoder_input_ids.shape[1] < max_length:
            # Get draft tokens autoregressively
            draft_tokens, draft_probs = self.get_draft_logits(
                encoder_inputs.input_ids,
                encoder_inputs.attention_mask,
                decoder_input_ids,
                self.gamma
            )

            draft_tokens = torch.tensor(draft_tokens, device=self.device)
            draft_probs = torch.tensor(draft_probs, device=self.device)

            if len(draft_tokens) == 0:
                raise ValueError("Draft tokens not generated.")

            # Get target probabilities in parallel
            # start = time.time()
            target_probs, target_logits = self.get_target_probs(
                encoder_inputs.input_ids,
                encoder_inputs.attention_mask,
                decoder_input_ids,
                draft_tokens
            )
            # print("Time taken for target probs: ", time.time() - start)

            # Verify tokens
            n_accepted = self.verify_tokens(target_probs, draft_tokens, draft_probs, self.temperature)
            # print(n_accepted)

            # Accept verified tokens
            if n_accepted > 0:
                decoder_input_ids = torch.cat([
                    decoder_input_ids,
                    draft_tokens[:n_accepted].unsqueeze(0)
                ], dim=1)
                
            with torch.no_grad():
                n_rejected = self.gamma - n_accepted
                total_tokens += self.gamma
                accepted_tokens += n_accepted
                if n_rejected > 0:
                    probs = target_logits[:, -n_rejected, :] #- draft_logits[:, 1-n_rejected, :]
                else:
                    probs = target_logits[:, -1, :]
                    
                probs = F.softmax(probs, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1)

                decoder_input_ids = torch.cat([decoder_input_ids, token_id], dim=1)

            # Check for end of sequence
            if decoder_input_ids[0][-1].item() == self.tokenizer.eos_token_id:
                break

        # Decode translation
        translation = self.tokenizer.decode(
            decoder_input_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        perc_accepted = accepted_tokens / total_tokens * 100
        return translation, perc_accepted
    
class NormalDecoder:
    def __init__(
        self,
        model_name: str = "google-t5/t5-3b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto')
        self.model.eval()

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Get logits from model for the last token."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
            return outputs.logits[:, -1, :]

    def sample_token(self, logits: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a token from logits using temperature sampling."""
        if temperature == 0:
            # Greedy sampling
            token_id = torch.argmax(logits, dim=-1)
            prob = torch.ones_like(token_id, dtype=torch.float)
        else:
            # Temperature sampling
            probs = F.softmax(logits / temperature, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            prob = probs.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
        return token_id, prob

    def translate(
        self,
        source_text: str,
        max_length: int = 128,
        temperature: float = 0.7
    ) -> str:
        """Translate source text using the normal T5 model."""
        # Encode source text
        encoder_inputs = self.tokenizer(
            f"translate English to German: {source_text}",
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Initialize decoder input with start token
        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]], device=self.device)

        while decoder_input_ids.shape[1] < max_length:
            # Generate logits for the next token
            logits = self.get_logits(
                encoder_inputs.input_ids,
                encoder_inputs.attention_mask,
                decoder_input_ids
            )

            # Sample a token
            token_id, _ = self.sample_token(logits, temperature)

            # Add token to the decoder input
            decoder_input_ids = torch.cat(
                [decoder_input_ids, token_id.view(1, 1)],
                dim=1
            )

            # Break if end token is generated
            if token_id.item() == self.tokenizer.eos_token_id:
                break

        # Decode and return translation
        translation = self.tokenizer.decode(
            decoder_input_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return translation

# Initialize decoder
from tqdm import tqdm

speculative_decoder = SpeculativeDecoder(gamma=4, temperature=0.1)
normal_decoder = NormalDecoder()

spec_total_time = 0
normal_total_time = 0
total_iters = 0

for i in tqdm(en_gr_dataset['test']['translation'][:30]):
    source_text = i['en']
    target_text = i['de']
    
    # Time the translation
    start_time = time.time()
    spec_translation, pc = speculative_decoder.translate(source_text)
    end_time = time.time()

    spec_time = end_time - start_time

    # spec_total_time += spec_time
    
    start_time = time.time()
    normal_translation = normal_decoder.translate(source_text)
    end_time = time.time()

    normal_time = end_time - start_time

    # normal_total_time += normal_time

    print(f"Source: {source_text}")
    print(f"Normal Translation: {normal_translation}")
    print(f"Time taken: {normal_time:.2f} seconds")
    print(f"Speculative Translation: {spec_translation}")
    print(f"Time taken: {spec_time:.2f} seconds")
    print(f"Percentage tokens accepted: {pc:.2f}%")
    
    print(f"Target: {target_text}")

    if normal_time - spec_time > -0.1:
        spec_total_time += spec_time
        normal_total_time += normal_time
        total_iters += 1

print(f"\nAverage time taken for normal decoding: {normal_total_time / total_iters:.2f} seconds")
print(f"Average time taken for speculative decoding: {spec_total_time / total_iters:.2f} seconds")
print(f"Average speedup over {total_iters} iterations: {normal_total_time / spec_total_time:.2f}x")