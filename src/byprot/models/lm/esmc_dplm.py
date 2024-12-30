import torch
import torch.nn as nn
from byprot.models import register_model
from tqdm import tqdm
from esm.models.esmc import ESMC
import math

from esm.tokenization import EsmSequenceTokenizer
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]

@register_model('mlm_esmc')
class EsmcForDPLM(nn.Module):
    """
    ESMC model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        n_layers (int): The number of transformer layers.
    """

    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.esmc = ESMC.from_pretrained(cfg.net.name,device=torch.device('cpu'))
        tokenizer = EsmSequenceTokenizer()
        token_to_id = {tok: ind for ind, tok in enumerate(SEQUENCE_VOCAB)}
        self.tokenizer = tokenizer

        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.x_id = token_to_id['X']
        
        self.contact_head = None
        self.tokenizer = tokenizer
        
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        **kwargs
    ):
        """
        Performs forward pass through the ESMC model. Check utils to see how to tokenize inputs from raw data.

        Args:
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            ESMCOutput: The output of the ESMC model.

        """
        # For EMSC, a boolean mask is created in place of sequence_id if not specified.
        sequence_id = input_ids != self.tokenizer.pad_token_id
        output = self.esmc(input_ids, sequence_id=sequence_id)
        output = {
            "logits": output.sequence_logits,
            "embeddings": output.embeddings,
            "last_hidden_state": output.hidden_states[-1]
        }
        return output
   
    def forward_encoder(self, batch, **kwargs):
        return {}
    
    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) &
            output_tokens.ne(self.bos_id) &
            output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= (~partial_masks)
        return non_special_sym_mask
    
    def initialize_output_tokens(self, batch, encoder_out, partial_masks=None, **kwargs):
        tokens = batch['input_ids']
        if tokens is None:
            raise NotImplementedError
        else:
            output_mask = self.get_non_special_sym_mask(tokens, partial_masks=partial_masks)

            output_tokens = tokens.masked_fill(output_mask, self.mask_id)
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

            return output_tokens, output_scores
        
    def forward_decoder(self, prev_decoder_out, encoder_out, need_attn_weights=False, partial_masks=None,
                        sampling_strategy='argmax'):
        output_tokens = prev_decoder_out['output_tokens'].clone()
        output_scores = prev_decoder_out['output_scores'].clone()
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        output_masks = self.get_non_special_sym_mask(output_tokens, partial_masks=partial_masks)

        esm_out = self.forward(
            input_ids=output_tokens,
        )
        logits = esm_out['logits']

        logits[..., self.mask_id] = -math.inf
        logits[..., self.x_id] = -math.inf
        
        if sampling_strategy == 'argmax':
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == 'sample':
            _tokens, _scores = sample_from_categorical(logits, temperature=temperature)
        
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            step=step + 1,
            max_step=max_step,
            history=history,
        )
        
    def generate(self, batch, tokenizer=None, 
                 max_iter=None, temperature=None, 
                 partial_masks=None,
                 sampling_strategy='gumbel_argmax'):
        tokenizer = tokenizer 
        max_iter = max_iter
        temperature = temperature

        # 0) encoding
        encoder_out = self.forward_encoder(batch)
        # 1) initialized from all mask tokens
        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(
            batch, encoder_out=encoder_out, partial_masks=partial_masks)
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
        )

        prev_decoder_out['output_masks'] = self.get_non_special_sym_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )

        for step in tqdm(range(max_iter), desc='Decoding'):
            # predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    encoder_out=encoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy
                )

            output_tokens = decoder_out['output_tokens']
            output_scores = decoder_out['output_scores']

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out['history']
            )

        decoder_out = prev_decoder_out
        return decoder_out['output_tokens'], decoder_out['output_scores']
    

def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores