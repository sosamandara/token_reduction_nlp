import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2MLP, Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer, BaseModelOutputWithPastAndCrossAttentions, logger, add_start_docstrings,add_start_docstrings_to_model_forward
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import time

@dataclass
class CustomModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, k_percent=0.1, selection_method="top_k"):
        super().__init__(config)
        self.k_percent = k_percent  # Adjustable top k% of tokens to keep
        self.selection_method = selection_method  # Method for selecting tokens to remove

    def custom_attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        batch_size, num_heads, query_length, key_length = attn_weights.size()
        attn_weights_trial = nn.functional.softmax(attn_weights, dim=-1)

        mean_attention_scores = attn_weights_trial.mean(dim=1)  # Shape: (batch_size, seq_len, seq_len)
        #print(mean_attention_scores.shape)
        mean_attention_scores = mean_attention_scores.mean(dim=-2)
        #print(mean_attention_scores)
        k = max(0, int(key_length * self.k_percent))

        if self.selection_method == "top_k":
            bottom_k_indices = torch.topk(mean_attention_scores, k, largest=False).indices
        elif self.selection_method == "random":
            indices = torch.arange(key_length)
            bottom_k_indices = indices[torch.randperm(key_length)[:k]]
        elif self.selection_method == "variance":
            variance = torch.var(attn_weights, dim=1).mean(dim=-2)
            bottom_k_indices = torch.topk(variance, k, largest=False).indices
        elif self.selection_method == "sum_attention":
            sum_attention_scores = attn_weights_trial.sum(dim=1).mean(dim=-2)
            bottom_k_indices = torch.topk(sum_attention_scores, k, largest=False).indices
        elif self.selection_method == "max_attention":
            max_attention_scores = attn_weights_trial.max(dim=1).values.mean(dim=-2)
            bottom_k_indices = torch.topk(max_attention_scores, k, largest=False).indices
        elif self.selection_method == "weighted_sampling":
            probabilities = 1.0 - mean_attention_scores
            probabilities = probabilities / probabilities.sum()
            bottom_k_indices = torch.multinomial(probabilities, k, replacement=False)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        # Debugging: Print the value of k and bottom_k_indices
        #print(f"Debug: k value: {k}, bottom_k_indices shape: {bottom_k_indices.shape}")
        #print(bottom_k_indices)
        
        #-----------
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        input_tokens = tokenizer.convert_ids_to_tokens(torch.arange(key_length))
        removed_tokens = tokenizer.convert_ids_to_tokens(bottom_k_indices)

        print(f"Layer: {self.layer_idx}")  # Assuming you have `self.layer_idx` defined
        print("Input tokens:", input_tokens)
        print("Removed tokens:", removed_tokens)
        #----------


        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            #print("attn_weights",attn_weights,attn_weights.shape)
            batch_size, num_heads, query_length, key_length = attn_weights.size()
            attention_mask = attention_mask[:,:key_length]
            #print("attention_mask",attention_mask,attention_mask.shape)
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights, bottom_k_indices, key_length

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
          past_key, past_value = layer_past
          key = torch.cat([past_key, key], dim=-1)
          value = torch.cat([past_value, value], dim=-1)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights, bottom_k_indices, key_length = self.custom_attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        mask = torch.ones(key_length, dtype=torch.bool, device=hidden_states.device)
        mask[bottom_k_indices] = False

        ## Debugging: Print the mask
        #print(mask)
#
        ## Update tensors to reflect the removal of tokens
        #print(query.shape)
        #print(key.shape)
        #print(value.shape)
        #query = query[:, :, mask, :]
        #key = key[:, :, mask, :]
        #value = value[:, :, mask, :]
        #attn_scores = attn_scores[:, :, mask, :][:, :, :, mask]
#
        ##if attention_mask is not None:
        ##    attention_mask = attention_mask[:, :, :, mask]
        #    #attention_mask = attention_mask[:, :, mask, :]
#
        #attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        #attn_output = torch.matmul(attn_weights, value)
        #attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        #attn_output = self.c_proj(attn_output)
        #attn_output = self.resid_dropout(attn_output)
#
        #present = (key, value) if use_cache else None

        return attn_output, mask#present, mask

GPT2_ATTENTION_CLASSES = {
    "eager": CustomGPT2Attention,
    #"flash_attention_2": GPT2FlashAttention2,
}

class CustomGPT2Block(GPT2Block):
    def __init__(self, config, k_percent=0.1, selection_method="top_k", apply_custom_attention=False):
        super().__init__(config)
        self.apply_custom_attention = apply_custom_attention
        if apply_custom_attention:
            self.attn = CustomGPT2Attention(config, k_percent=k_percent, selection_method=selection_method)
        #attention_class = GPT2_ATTENTION_CLASSES[config._attn_implementation]

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        #print(f"Block input shape: {hidden_states.shape}")
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        # Get mask and adjust tensors from custom attention
        if self.apply_custom_attention:
            #print("Customized Attention",self.apply_custom_attention)
            attn_outputs, mask = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            #hidden_states = attn_output
            #print("hidden_states",hidden_states,hidden_states.shape)
            # Remove the same tokens from the residual to match the dimension
            #print("residual",residual,residual.shape)
            #print("attn_output",attn_output,attn_output.shape)
            residual = residual[:, mask, :]
            attn_outputs = attn_outputs[:, mask, :]
            #print("residual",residual,residual.shape)
            #print("attn_output",attn_output,attn_output.shape)
            #print("attn_outputs[0]",attn_output[0],attn_output[0].shape)
            #print("attn_outputs[1:]",attn_output[1:],attn_output[1:].shape)
            #print("attn_outputs[0]+residual",attn_output[0]+residual,(attn_output[0]+residual).shape)
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            outputs = attn_outputs[1:]
            #print("outputs", outputs.shape)
            # residual connection
            hidden_states = attn_output + residual

            #if attention_mask is not None:
            #    attention_mask = attention_mask[:, :, :, mask]
                #attention_mask = attention_mask[:, :, mask, :]

            if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
              if not hasattr(self, "crossattention"):
                  raise ValueError(
                      f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                      "cross-attention layers by setting `config.add_cross_attention=True`"
                  )
              residual = hidden_states
              hidden_states = self.ln_cross_attn(hidden_states)
              cross_attn_outputs, mask = self.crossattention(
                  hidden_states,
                  attention_mask=attention_mask,
                  head_mask=head_mask,
                  encoder_hidden_states=encoder_hidden_states,
                  encoder_attention_mask=encoder_attention_mask,
                  output_attentions=output_attentions,
              )
              attn_output = cross_attn_outputs[0]
              # residual connection
              hidden_states = residual + attn_output
              outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

            if layer_past is not None:
                past_key, past_value = layer_past
                #print("past_key",past_key,past_key.shape)
                #print("past_value",past_value,past_value.shape)
                #print("layer_past",layer_past,layer_past.shape)
                past_key = past_key[:, :, mask]
                past_value = past_value[:, :, mask]
                layer_past = (past_key, past_value)

        else:
            #print("Normal Attention",self.apply_custom_attention)
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]
            outputs = attn_outputs[1] if use_cache else None

        residual = hidden_states
        #print("hidden_states",hidden_states,hidden_states.shape)
        hidden_states = self.ln_2(hidden_states)
        #print("hidden_states",hidden_states,hidden_states.shape)

        feed_forward_hidden_states = self.mlp(hidden_states)
        #print("feed_forward_hidden_states",feed_forward_hidden_states,feed_forward_hidden_states.shape)
        hidden_states = residual + feed_forward_hidden_states
        #print("hidden_states",hidden_states,hidden_states.shape)

        #print(f"Block output shape: {hidden_states.shape}")
        #outputs = (hidden_states,)
        #if use_cache:
        #    outputs += (present,)
        #if output_attentions:
        #    outputs += (attn_outputs,)
        #print("hidden_states", hidden_states, hidden_states.shape, type(hidden_states))
        #print("outputs", outputs, outputs.shape, type(outputs))
        if use_cache:
            #print(len((hidden_states,)),(hidden_states,), type((hidden_states,)))
            outputs = (hidden_states,) + (outputs,)
        else:
            outputs = (hidden_states,) + (outputs[1:],)

        return outputs

class CustomGPT2Model(GPT2Model):
    def __init__(self, config, k_percent=0.1, selection_method="top_k", layers_to_prune=None):
        super().__init__(config)
        if layers_to_prune is None:
            layers_to_prune = [0, 1, 2]  # Apply custom attention only to the first few layers by default
        self.h = nn.ModuleList([CustomGPT2Block(config, k_percent=k_percent[i], selection_method=selection_method, apply_custom_attention=(i in layers_to_prune)) for i in range(config.n_layer)])
        #self.h = nn.ModuleList([CustomGPT2Block(config, k_percent=layers_to_prune.get(i, None)) for i in range(config.n_layer)])

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, k_percent=0.1, selection_method="top_k", layers_to_prune=None):
        super().__init__(config)
        self.transformer = CustomGPT2Model(config, k_percent=k_percent, selection_method=selection_method, layers_to_prune=layers_to_prune)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size = input_shape[0]

        if past_key_values is None:
            past_key_values = [None] * len(self.transformer.h)

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.transformer.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.transformer.drop(hidden_states)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        seq_lengths = torch.full((batch_size,), input_shape[-1], dtype=torch.long, device=hidden_states.device)
        for i, (block, layer_past) in enumerate(zip(self.transformer.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            #print(f"Layer {i} input shape: {hidden_states.shape}")
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            #print(f"Layer {i} output shape: {hidden_states.shape}")
            if use_cache:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

            if hasattr(outputs, "mask"):
                seq_lengths = seq_lengths - (~outputs.mask).sum(dim=-1)

        hidden_states = self.transformer.ln_f(hidden_states)
        output_shape = (-1, hidden_states.size(1), hidden_states.size(-1))
        #print("Final Output Shape:", output_shape)
        #print("Final Hidden States Shape:", hidden_states.shape)
        #hidden_states = hidden_states.view(output_shape)
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (lm_logits,) + presents + (all_hidden_states,) + (all_self_attentions,) + (all_cross_attentions,)
            return output

        return CustomModelOutput(
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )