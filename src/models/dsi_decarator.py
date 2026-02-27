# !/usr/bin/env python
# -*-coding:utf-8 -*-
import os
import torch
from datetime import datetime
from transformers import AutoTokenizer
import json
import functools
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast  # 对model output进行管理。
from transformers.cache_utils import Cache


def enable_dsi(model_class):
    """
    Decorator that adds DSI (Dual-Stream Internalization Test-time Training) functionality to any CausalLM model class.

    Args:
        model_class: A Transformers CausalLM model class (e.g., LlamaForCausalLM, GPT2LMHeadModel, etc.)
                    or AutoModelForCausalLM factory class

    Returns:
        Enhanced model class with DSI capabilities
    """

    if not hasattr(model_class, 'forward'):
        return _create_dsi_auto_model_class(model_class)

    # Store original methods for regular model classes
    original_init = model_class.__init__
    original_forward = model_class.forward

    def enhanced_init(self, config, *args, **kwargs):
        """Enhanced __init__ that adds DSI attributes"""
        # Call original __init__
        original_init(self, config, *args, **kwargs)

        # Initialize DSI-specific attributes
        self.delta = None
        self.high_entropy_detected = False
        self.high_entropy_position = None
        self.entropy_threshold = None
        self.entropy_history = []
        self.long_entropy_history = []

        self.index = None
        self.prompt_only = False  # 是否进行tuning

    def enhanced_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            masked_token_ids: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        struc_dataset, con_query_mask, ttl_state = self.model.struc_dataset, self.model.con_query_mask, self.model.ttl_state

        # Handle default values like in original implementation
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call the base model (not the full forward) - this is the key difference!
        outputs = self.model(
            input_ids=input_ids,  # B,T
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        # Extract hidden states - consistent with original implementation
        hidden_states = outputs[0]  # B, T, D

        # Store original hidden states for entropy comparison
        original_hidden_states = hidden_states.clone()

        # structure hidden
        struc_outputs = []
        for struc in struc_dataset:
            struc_input_ids, struc_attention_mask = struc['input_ids'], struc['attention_mask']
            struc_output = self.model(
                input_ids=struc_input_ids,
                attention_mask=struc_attention_mask
            )
            struc_hidden_states = struc_output[0]  # B, T, D
            struc_outputs.append({'struc_hidden': struc_hidden_states.clone(), 'struc_label': struc['labels'].clone()})

        prompt_only = self.prompt_only
        stage = "prompt" if prompt_only else "generation"

        # Apply DSI logic, new hidden states, 还存了response的entropy分析，是对整体response
        if ttl_state == 'online':
            hidden_states = apply_dsi_logic(
                self,
                hidden_states,
                input_ids,
                masked_token_ids,
                prompt_only,
                struc_outputs,
                con_query_mask
            )
        elif ttl_state == 'offline':
            hidden_states = apply_dsi_logic_off(
                self,
                hidden_states,
                input_ids,
                masked_token_ids,
                prompt_only,
                struc_outputs,
                con_query_mask
            )

        
        handle_entropy_analysis(
            self,
            original_hidden_states,
            hidden_states,
            input_ids,
            logits_to_keep
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # B,W,V

        
        logits = apply_entropy_control(
            self,
            logits,
            past_key_values,
            input_ids,
            logits_to_keep,
            stage
        )

        # Handle loss computation - exactly like original implementation
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        # Return in the same format as original implementation
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    # Replace methods in the class using module-level functions
    model_class.__init__ = enhanced_init
    model_class.forward = enhanced_forward
    model_class.reset_entropy_detection = _reset_entropy_detection_method
    model_class.reset_model_parameters = _reset_model_parameters_method
    model_class._safe_decode_token = _safe_decode_token_method
    model_class._safe_decode_sequence = _safe_decode_sequence_method
    model_class.reset_offline_parameters = _reset_offline_parameters_method

    return model_class


def _create_dsi_auto_model_class(auto_model_class):
    """
    Create a DSI-enabled wrapper for AutoModelForCausalLM factory class.

    Args:
        auto_model_class: AutoModelForCausalLM class

    Returns:
        DSI-enabled wrapper class
    """

    class DSIAutoModelForCausalLM:
        """DSI-enabled wrapper for AutoModelForCausalLM"""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Load model and apply DSI functionality"""
            # Load the actual model using the original AutoModelForCausalLM
            model = auto_model_class.from_pretrained(*args,
                                                     **kwargs)  # dict_keys(['dtype', '_attn_implementation', 'device_map', 'trust_remote_code'])

            # Apply DSI functionality to the loaded model instance, add several functions and change several functions
            model = _apply_dsi_to_instance(model)

            return model

        # Forward other class methods to the original class
        def __getattr__(self, name):
            return getattr(auto_model_class, name)

    return DSIAutoModelForCausalLM


def _apply_dsi_to_instance(model):
    """
    Apply DSI functionality to an already instantiated model.

    Args:
        model: An instantiated CausalLM model

    Returns:
        The same model with DSI functionality added
    """

    # Initialize DSI-specific attributes, 都是全局参数
    model.delta = None
    model.high_entropy_detected = False
    model.high_entropy_position = None
    model.entropy_threshold = None
    model.entropy_history = []
    model.long_entropy_history = []

    model.index = None
    model.prompt_only = False

    # Replace the model's forward method with our enhanced version
    # Use functools.partial to create a pickleable bound method
    model.forward = functools.partial(_enhanced_forward_for_instance,
                                      model)

    # Add the missing DSI methods
    _add_dsi_methods(model)

    return model


def _enhanced_forward_for_instance(
        model,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        masked_token_ids: Optional[torch.LongTensor] = None,
        **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """Enhanced forward method with DSI functionality for model instances， key"""
    # Handle default values like in original implementation
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else model.config.use_cache
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    # Prepare arguments for original forward method
    forward_kwargs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'past_key_values': past_key_values,
        'inputs_embeds': inputs_embeds,
        'use_cache': use_cache,
        'output_attentions': output_attentions,
        'output_hidden_states': output_hidden_states,
        'return_dict': return_dict,
        **kwargs,
    }

    struc_dataset, con_query_mask, ttl_state = model.struc_dataset, model.con_query_mask, model.ttl_state

    # Add cache_position if supported by the model
    import inspect
    original_forward = getattr(model.__class__, 'forward', None)
    if original_forward and 'cache_position' in inspect.signature(original_forward).parameters:
        forward_kwargs['cache_position'] = cache_position

    # Call the underlying model's forward method (self.model for CausalLM models)
    outputs = model.model(**forward_kwargs)

    # Extract hidden states - consistent with original implementation
    hidden_states = outputs[0]
    original_hidden_states = hidden_states.clone()

    # structure hidden
    struc_outputs = []
    for i, struc in enumerate(struc_dataset):
        struc_input_ids, struc_attention_mask = struc['input_ids'], struc['attention_mask']
  
        struc_output = model.model(
            input_ids=struc_input_ids,
            attention_mask=struc_attention_mask
        )
        struc_hidden_states = struc_output[0]
        struc_outputs.append({'struc_hidden': struc_hidden_states.clone(), 'struc_label': struc['labels'].clone()})

    prompt_only = model.prompt_only
    stage = "prompt" if prompt_only else "generation"

    # Apply DSI logic
    if ttl_state == 'online':
        hidden_states = apply_dsi_logic(model, hidden_states, input_ids, masked_token_ids, prompt_only, struc_outputs,
                                        con_query_mask)
    else:
        hidden_states = apply_dsi_logic_off(model, hidden_states, input_ids, masked_token_ids, prompt_only,
                                            struc_outputs, con_query_mask)
    # Handle entropy analysis and recording
    handle_entropy_analysis(model, original_hidden_states, hidden_states, input_ids,
                            logits_to_keep)

    # Recompute logits with modified hidden states
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = model.lm_head(hidden_states[:, slice_indices, :])  # 是一个范围，新生成的内容

    # Apply entropy-based early stopping if enabled
    logits = apply_entropy_control(
        model,
        logits,
        past_key_values,
        input_ids,
        logits_to_keep,
        stage
    )
    # Handle loss computation - exactly like original implementation
    loss = None
    if labels is not None:
        loss = model.loss_function(logits=logits, labels=labels, vocab_size=model.config.vocab_size, **kwargs)

    # Return in the same format as original implementation
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
        hidden_states=outputs.hidden_states if output_hidden_states else None,
        attentions=outputs.attentions if output_attentions else None,
    )


def cal_struc_loss(struc_outputs, model, loss_fct, delta_lis, masked_token_ids=None):
    """
    :param struc_outputs: [{'struc_hidden': , 'struc_label': }]
    :param model:
    :param loss_fct:
    :param transformed_hidden:
    :return:
    """
    struc_loss_lis = []
    for struc in struc_outputs:
        struc_hidden, struc_label = struc['struc_hidden'], struc['struc_label']
        transformed_struc_hidden = struc_hidden + delta_lis[0] + delta_lis[1]  # B,T,D
        logits_struc = model.lm_head(transformed_struc_hidden)  # B,T,V

        # 1. shift
        shift_logits = logits_struc[..., :-1, :].contiguous()  # # B,T-1,V
        struc_label = struc_label.contiguous().to(logits_struc.device)  # B, T
        shift_labels = struc_label[:, 1:].contiguous()

        filtered_labels = shift_labels[0]
        filtered_labels = filtered_labels[filtered_labels != -100]
        

        # 2. mask
        masked_labels = shift_labels.clone()
        if masked_token_ids and len(masked_token_ids) > 0:
            mask = torch.isin(masked_labels, torch.tensor(masked_token_ids, device=model.device))
            masked_labels[mask] = -100
   
        # 3. reshape
        logits_struc = shift_logits.reshape(-1, shift_logits.size(-1))
        struc_labels_flat = masked_labels.reshape(-1)

        # 4. 如果无效位置，则跳过该样本
        if struc_labels_flat.eq(-100).all():
            print("No valid place for label!")
            continue

        struc_loss = loss_fct(logits_struc, struc_labels_flat)

        # 5. cal loss
        struc_loss_lis.append(struc_loss)

    if len(struc_loss_lis) == 0:
        return torch.tensor(0.0).to(model.device)
    return sum(struc_loss_lis) / len(struc_loss_lis)


def apply_dsi_logic(model, hidden_states, input_ids, masked_token_ids, prompt_only, struc_outputs, con_query_mask,
                    logits_to_keep=None):
    """
    Apply DSI (Test-time Training) logic to hidden states; 核心!!!

    Args:
        model: The model instance
        hidden_states: Current hidden states
        input_ids: Input token IDs
        masked_token_ids: Token IDs to mask during training

    Returns:
        Modified hidden states after DSI processing
    """
    # 返回前添加维度检查
    if hidden_states.shape[1] == 0:  # 序列长度维度为0
        print(f"错误: hidden_states序列长度为0! shape={hidden_states.shape}")

    if prompt_only:
        if model.delta is not None:
            
            pass
        times = int(os.environ.get("times", 1))  # update times
        lr = float(os.environ.get("lr", 0.1))

        loss_list = []
        with torch.enable_grad():

            if model.delta is not None:  # 之前最开始的初始化，就在下面normal
                # print("Formal Optimization")
                delta_high = nn.Parameter(0.0 * torch.randn([1, 1, hidden_states.shape[-1]]).to(
                    hidden_states))
                delta_stru = nn.Parameter(0.0 * torch.randn([1, 1, hidden_states.shape[-1]]).to(
                    hidden_states))
                delta_query = nn.Parameter(0.0 * torch.randn([1, 1, hidden_states.shape[-1]]).to(
                    hidden_states))
                optimizer_high = torch.optim.AdamW([delta_high], lr=lr, weight_decay=1e-8, eps=1e-5)
                optimizer_stru = torch.optim.AdamW([delta_stru], lr=lr, weight_decay=1e-8, eps=1e-5)  # 感觉不要也行，就是做多任务去
                optimizer_query = torch.optim.AdamW([delta_query], lr=lr, weight_decay=1e-8, eps=1e-5)

                loss_fct = nn.CrossEntropyLoss()

                for _ in range(times):
                    optimizer_high.zero_grad()
                    optimizer_stru.zero_grad()
                    optimizer_query.zero_grad()
                    transformed_hidden = hidden_states + delta_high + delta_stru
                    transformed_hidden = transformed_hidden + delta_query * con_query_mask.unsqueeze(
                        -1)
                    logits = model.lm_head(transformed_hidden)  # B,T,V
                    shift_logits = logits[..., :-1, :].contiguous()

                    shift_labels = input_ids[:, 1:].contiguous()  # B, T-1
                    shift_labels = shift_labels.to(shift_logits.device)

                    # Apply the mask to the labels
                    masked_labels = shift_labels.clone()
                    if masked_token_ids is not None:
                        for token_id in masked_token_ids:
                            masked_labels[masked_labels == token_id] = -100
                   
                    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                       masked_labels.view(-1))

                    # query_ce_loss
                    query_weight = float(os.environ.get("query_weight", "0.1"))

                    query_mask = con_query_mask[:, 1:].contiguous()  # B,T-1
                    query_masked_labels = masked_labels.clone()  # B,T-1
                    # 应该是替换为-100
                    query_masked_labels = query_masked_labels * query_mask + (
                                1 - query_mask) * -100  # 只保留query之后的token loss; B,T-1;

                    # query logits masked
                    query_masked_logits = shift_logits  # B, T-1, V
                    query_ce_loss = loss_fct(query_masked_logits.view(-1, query_masked_logits.size(-1)),
                                             query_masked_labels.view(-1))

           
                    # structure labels
                    structure_weight = float(os.environ.get("structure_weight", "0.1"))
                    ce_loss_stru = cal_struc_loss(struc_outputs, model, loss_fct, [delta_stru, delta_high],
                                                  masked_token_ids)

                   
                    entropy, entropy_loss = cal_uncertainty(logits, uncertain_name='entropy')

                    # Combine losses using weighted average
                    entropy_weight = float(os.environ.get("entropy_weight", "0.1"))

                    loss = ce_loss + entropy_weight * entropy_loss + structure_weight * ce_loss_stru + query_weight * query_ce_loss

                    loss_list.append(
                        {
                            "ce_loss": ce_loss.item(),
                            "entropy_loss": entropy_loss.item(),
                            "query_ce_loss": query_ce_loss.item(),
                            "stru_loss": ce_loss_stru.item(),
                            "loss": loss.item()
                        }
                    )
                    print(f"Append: {loss_list[-1]}")

                    loss.backward()
                 
                    optimizer_high.step()
                    optimizer_stru.step()
                    optimizer_query.step()

                
                hidden_states = hidden_states + delta_high + delta_stru + delta_query * con_query_mask.unsqueeze(-1)

                if response_entropy_file := os.environ.get("response_entropy_file_after", ""):
                    
                    _record_high_entropy_token(model, model.lm_head(hidden_states), logits_to_keep,
                                               response_entropy_file)

    
            else:

                delta_normal = nn.Parameter(0.0 * torch.randn([1, 1, hidden_states.shape[-1]]).to(hidden_states))
                optimizer_normal = torch.optim.AdamW([delta_normal], lr=lr, weight_decay=1e-8, eps=1e-5)
                for _ in range(times):
                    optimizer_normal.zero_grad()
                    transformed_hidden = hidden_states + delta_normal
                    logits = model.lm_head(transformed_hidden)
                    loss_fct = nn.CrossEntropyLoss()
                    shift_logits = logits[..., :-1, :].contiguous()

                    # Use prompt as labels
                    shift_labels = input_ids[:, 1:].contiguous()
                    shift_labels = shift_labels.to(shift_logits.device)

                    # Apply the mask to the labels
                    masked_labels = shift_labels.clone()
                    if masked_token_ids and len(masked_token_ids) > 0:
                        mask = torch.isin(masked_labels, torch.tensor(masked_token_ids, device=masked_labels.device))
                        masked_labels[mask] = -100

                    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), masked_labels.view(-1))

                    # Only use cross-entropy loss for delta_normal
                    loss = ce_loss

                    loss.backward()
                    optimizer_normal.step()

                # Store delta_normal for subsequent generation stages
                model.delta = delta_normal.detach().clone()


        model.prompt_only = False
        torch.cuda.empty_cache()

    else:  # 直接生成，不动最后的hidden states
        if model.delta is not None:
            # Apply delta_normal (cross-entropy optimized) for generation
            # Note: In the original implementation, this was commented out
            # hidden_states = hidden_states + model.delta
            pass

    return hidden_states


def cal_uncertainty(logits, uncertain_name='entropy'):
    if uncertain_name == 'entropy':
        last_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        temperature = float(os.environ.get("temperature", "1.0"))
        last_probs = F.softmax(last_logits / temperature, dim=-1)
        entropy = -torch.sum(last_probs * torch.log(last_probs + 1e-10), dim=-1)  # Shape: [batch_size]
        entropy_loss = torch.mean(entropy)  # Average over batch
        return entropy, entropy_loss
    elif uncertain_name == 'perplexity':
        last_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        temperature = float(os.environ.get("temperature", "1.0"))
        last_probs = F.softmax(last_logits / temperature, dim=-1)
        entropy = -torch.sum(last_probs * torch.log(last_probs + 1e-10), dim=-1)  # Shape: [batch_size]
        perplexity = torch.exp(entropy)
        perplexity_loss = torch.mean(perplexity)  # Average over batch
        return perplexity, perplexity_loss
    elif uncertain_name == 'energy':
        last_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        temperature = float(os.environ.get("temperature", "1.0"))
        logits_scaled = last_logits / temperature
        energy = -1 * torch.logsumexp(logits_scaled, dim=-1)  # [batch_size]
        uncertainty_energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-10)
        avg_uncertainty_energy = torch.mean(uncertainty_energy)
        return uncertainty_energy, avg_uncertainty_energy
    elif uncertain_name == 'confidence_gap':
        last_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        temperature = float(os.environ.get("temperature", "1.0"))
        last_probs = F.softmax(last_logits / temperature, dim=-1)
        top2_probs = torch.topk(last_probs, k=2, dim=-1)[0]  # [batch_size, 2]
        margin = top2_probs[:, 0] - top2_probs[:, 1]  # [batch_size]，最大 - 第二大
        uncertainty_margin = 1 - margin
        avg_uncertainty_margin = torch.mean(uncertainty_margin)
        return uncertainty_margin, avg_uncertainty_margin
    else:
        raise ValueError(f"Unknown uncertainty measure: {uncertain_name}")


def apply_dsi_logic_off(model, hidden_states, input_ids, masked_token_ids, prompt_only, struc_outputs, con_query_mask,
                        logits_to_keep=None):
    """
    Apply DSI (Test-time Training) logic to hidden states; 核心!!!

    Args:
        model: The model instance
        hidden_states: Current hidden states
        input_ids: Input token IDs
        masked_token_ids: Token IDs to mask during training

    Returns:
        Modified hidden states after DSI processing
    """

    if prompt_only:
        # print("XXXXX1", model.delta)
        if model.delta is not None:
            
            pass
        times = int(os.environ.get("times", 1))  # update times
        lr = float(os.environ.get("lr", 0.1))

        loss_list = []
        with torch.enable_grad():

            if model.delta is not None:
                # print("Formal Optimization")
                if model.delta_high is None:
                    delta_high = nn.Parameter(0.0 * torch.randn([1, 1, hidden_states.shape[-1]]).to(
                        hidden_states))

                    delta_stru = nn.Parameter(0.0 * torch.randn([1, 1, hidden_states.shape[-1]]).to(
                        hidden_states))
                    delta_query = nn.Parameter(0.0 * torch.randn([1, 1, hidden_states.shape[-1]]).to(
                        hidden_states))
                else:
                    delta_high = nn.Parameter(model.delta_high.clone())
                    delta_stru = nn.Parameter(model.delta_stru.clone())
                    delta_query = nn.Parameter(model.delta_query.clone())
                # Optimize delta_high with joint loss (CE + entropy)
                optimizer_high = torch.optim.AdamW([delta_high], lr=lr, weight_decay=1e-8, eps=1e-5)
                optimizer_stru = torch.optim.AdamW([delta_stru], lr=lr, weight_decay=1e-8, eps=1e-5)
                optimizer_query = torch.optim.AdamW([delta_query], lr=lr, weight_decay=1e-8, eps=1e-5)

                con_query_mask_exp = con_query_mask.unsqueeze(-1)  # [B, T, 1] 广播用
                shift_query_mask = con_query_mask[:, 1:].contiguous()  # [B, T-1] query区域掩码
                loss_fct = nn.CrossEntropyLoss()

                for _ in range(times):
                    optimizer_high.zero_grad()
                    optimizer_stru.zero_grad()
                    optimizer_query.zero_grad()
                    transformed_hidden = hidden_states + delta_high + delta_stru
                    transformed_hidden = transformed_hidden + delta_query * con_query_mask_exp
                    logits = model.lm_head(transformed_hidden)  # B,T,V
                    shift_logits = logits[..., :-1, :].contiguous()  # 预测下一个token

                    # Use prompt as labels
                    shift_labels = input_ids[:, 1:].contiguous()
                    shift_labels = shift_labels.to(shift_logits.device)

                    masked_labels = shift_labels.clone()
       
                    if masked_token_ids and len(masked_token_ids) > 0:
                
                        mask = torch.isin(masked_labels, torch.tensor(masked_token_ids, device=masked_labels.device))
                        masked_labels[mask] = -100
   
                    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                       masked_labels.view(-1))

                    # query_ce_loss
                    query_weight = float(os.environ.get("query_weight", "0.1"))  # 平衡Structure Loss和原有Loss
                    query_mask = shift_query_mask
                    query_masked_labels = masked_labels.clone()
        
                    query_masked_labels = query_masked_labels * query_mask + (
                                1 - query_mask) * -100


                    # query logits masked
                    query_masked_logits = shift_logits  # B, T-1, V
                    query_ce_loss = loss_fct(query_masked_logits.view(-1, query_masked_logits.size(-1)),
                                             query_masked_labels.view(-1))

                    # structure labels
                    structure_weight = float(os.environ.get("structure_weight", "0.1"))  # 平衡Structure Loss和原有Loss
                    ce_loss_stru = cal_struc_loss(struc_outputs, model, loss_fct, [delta_stru, delta_high],
                                                  masked_token_ids)

                    entropy, entropy_loss = cal_uncertainty(logits, uncertain_name='entropy')

                    # Combine losses using weighted average
                    entropy_weight = float(os.environ.get("entropy_weight", "0.1"))
                    loss = (
                                       1 - entropy_weight) * ce_loss + entropy_weight * entropy_loss + structure_weight * ce_loss_stru + query_weight * query_ce_loss

                    loss_list.append(
                        {
                            "ce_loss": ce_loss.item(),
                            "entropy_loss": entropy_loss.item(),
                            "query_ce_loss": query_ce_loss.item(),
                            "stru_loss": ce_loss_stru.item(),
                            "loss": loss.item()
                        }
                    )
                    print(f"Sample loss for cur times: {loss_list[-1]}")

                    loss.backward()

                    optimizer_high.step()
                    optimizer_stru.step()
                    optimizer_query.step()

  
                hidden_states = hidden_states + delta_high + delta_stru + delta_query * con_query_mask.unsqueeze(-1)
                # 存储参数
                model.delta_high = delta_high.detach().clone()
                model.delta_stru = delta_stru.detach().clone()
                model.delta_query = delta_query.detach().clone()

                if response_entropy_file := os.environ.get("response_entropy_file_after", ""):
                    _record_high_entropy_token(model, model.lm_head(hidden_states), logits_to_keep,
                                               response_entropy_file)


            else:
                if model.delta_normal is None:
                    delta_normal = nn.Parameter(0.0 * torch.randn([1, 1, hidden_states.shape[-1]]).to(hidden_states))
                else:
                    print("Continue!")
                    delta_normal = nn.Parameter(model.delta_normal.clone())
                optimizer_normal = torch.optim.AdamW([delta_normal], lr=lr, weight_decay=1e-8, eps=1e-5)
                for _ in range(times):
                    optimizer_normal.zero_grad()
                    transformed_hidden = hidden_states + delta_normal
                    logits = model.lm_head(transformed_hidden)
                    loss_fct = nn.CrossEntropyLoss()
                    shift_logits = logits[..., :-1, :].contiguous()

                    # Use prompt as labels
                    shift_labels = input_ids[:, 1:].contiguous()
                    shift_labels = shift_labels.to(shift_logits.device)

                    # Apply the mask to the labels
                    masked_labels = shift_labels.clone()
 
                    if masked_token_ids and len(masked_token_ids) > 0:

                        mask = torch.isin(masked_labels, torch.tensor(masked_token_ids, device=masked_labels.device))
                        masked_labels[mask] = -100

                    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), masked_labels.view(-1))

                    loss = ce_loss

                    loss.backward()
                    optimizer_normal.step()

                # Store delta_normal for subsequent generation stages
                model.delta = delta_normal.detach().clone()
                model.delta_normal = delta_normal.detach().clone()

        model.prompt_only = False
        torch.cuda.empty_cache()



    else:
        if model.delta is not None:
            # Apply delta_normal (cross-entropy optimized) for generation
            # Note: In the original implementation, this was commented out
            # hidden_states = hidden_states + model.delta
            pass

    return hidden_states


def _record_high_entropy_token(model, original_logits, logits_to_keep, response_entropy_file):
    """Record entropy analysis for response tokens only"""
    try:
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = slice(None)  # All tokens

        with torch.no_grad():
            # Get lm_head from model to compute logits

            # Calculate probabilities and entropy
            original_probs = F.softmax(original_logits, dim=-1)

            # Calculate entropy: -sum(p * log(p))
            original_entropy = -torch.sum(original_probs * torch.log(original_probs + 1e-10), dim=-1)  # B,T

            # Get predicted tokens
            original_tokens = torch.argmax(original_logits, dim=-1)

            # Handle different tensor shapes
            if len(original_tokens.shape) == 1:
                # If 1D tensor, treat as single batch
                batch_size = 1
                seq_len = original_tokens.shape[0]
                original_tokens = original_tokens.unsqueeze(0)
                original_entropy = original_entropy.unsqueeze(0)
            elif len(original_tokens.shape) == 2:
                batch_size, seq_len = original_tokens.shape
            else:
                # Handle higher dimensions by flattening
                original_shape = original_tokens.shape
                original_tokens = original_tokens.view(-1, original_shape[-1])
                original_entropy = original_entropy.view(-1, original_shape[-1])
                batch_size, seq_len = original_tokens.shape

            # Prepare response data list,[{}], len=B*T
            response_data = []

            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    # Create record for each response token
                    record = {
                        "batch_idx": batch_idx,
                        "seq_idx": seq_idx,
                        "original_entropy": original_entropy[batch_idx, seq_idx].item(),
                        "original_token": original_tokens[batch_idx, seq_idx].item(),
                        "original_token_decoded": _safe_decode_token(original_tokens[batch_idx, seq_idx].item()),
                    }
                    response_data.append(record)

            # Read existing data if file exists
            existing_data = []
            if os.path.exists(response_entropy_file):
                try:
                    with open(response_entropy_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []

            # Extend existing data with new response data
            existing_data.extend(response_data)

            # Write updated data back to file
            with open(response_entropy_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        # Log error but don't interrupt the forward pass
        import traceback
        print(f"Error in response entropy analysis: {e}")
        print(f"Traceback: {traceback.format_exc()}")


def handle_entropy_analysis(model, original_hidden_states, modified_hidden_states, input_ids, logits_to_keep):
    """Handle entropy recording and analysis"""

    # Calculate entropy and record analysis if enabled
    if os.environ.get("record_entropy", "False") == "True" and model.delta is not None:
        _record_entropy_analysis(model, original_hidden_states, modified_hidden_states, input_ids, logits_to_keep)


def _record_entropy_analysis(model, original_hidden_states, modified_hidden_states, input_ids, logits_to_keep):
    """Record entropy analysis for tokens before and after applying delta"""
    try:
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep,
                                                                   int) else logits_to_keep  # slice(a, None) → [a:]

        with torch.no_grad():
            original_logits = model.lm_head(original_hidden_states[:, slice_indices, :])  # B,W
            modified_logits = model.lm_head(modified_hidden_states[:, slice_indices, :])  # B,W

            # Calculate probabilities and entropy
            temperature = float(os.environ.get("temperature", "1.0"))
            original_probs = F.softmax(original_logits / temperature, dim=-1)
            modified_probs = F.softmax(modified_logits / temperature, dim=-1)

            # Calculate entropy: -sum(p * log(p))
            original_entropy = -torch.sum(original_probs * torch.log(original_probs + 1e-10), dim=-1)
            modified_entropy = -torch.sum(modified_probs * torch.log(modified_probs + 1e-10), dim=-1)

            # Get predicted tokens
            original_tokens = torch.argmax(original_logits, dim=-1)
            modified_tokens = torch.argmax(modified_logits, dim=-1)

            # Process each batch and sequence position, B,W
            batch_size, seq_len = original_tokens.shape

            # Get output file path
            output_file = os.environ.get("entropy_output_file", "entropy_analysis.jsonl")

            # Prepare data for logging
            entropy_data = []

            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                        actual_seq_idx = input_ids.shape[1] - logits_to_keep + seq_idx
                    else:
                        actual_seq_idx = seq_idx

                    # Skip if out of bounds
                    if actual_seq_idx >= input_ids.shape[1]:
                        continue

                    # Get input token (the token that produced this prediction)
                    input_token = input_ids[batch_idx, actual_seq_idx].item()

                    record = {
                        "batch_idx": batch_idx,
                        "seq_idx": actual_seq_idx,
                        "input_token": input_token,
                        "input_token_decoded": model._safe_decode_token(input_token),
                        "original_predicted_token": original_tokens[batch_idx, seq_idx].item(),
                        "original_predicted_decoded": model._safe_decode_token(
                            original_tokens[batch_idx, seq_idx].item()),
                        "original_entropy": original_entropy[batch_idx, seq_idx].item(),
                        "modified_predicted_token": modified_tokens[batch_idx, seq_idx].item(),
                        "modified_predicted_decoded": model._safe_decode_token(
                            modified_tokens[batch_idx, seq_idx].item()),
                        "modified_entropy": modified_entropy[batch_idx, seq_idx].item(),
                        "entropy_diff": (
                                modified_entropy[batch_idx, seq_idx] - original_entropy[batch_idx, seq_idx]).item(),
                    }
                    entropy_data.append(record)

            # Write to file
            with open(output_file, "a", encoding="utf-8") as f:
                for record in entropy_data:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as e:
        # Log error but don't interrupt the forward pass
        print(f"Error in entropy analysis: {e}")


def _record_response_entropy(model, original_hidden_states, modified_hidden_states, input_ids, logits_to_keep,
                             response_entropy_file):
    """Record entropy analysis for response tokens only"""
    try:
        # Calculate logits for both original and modified hidden states
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = slice(None)  # All tokens

        with torch.no_grad():
            # Get lm_head from model to compute logits
            original_logits = model.lm_head(original_hidden_states[:, slice_indices, :])
            modified_logits = model.lm_head(modified_hidden_states[:, slice_indices, :])

            # Calculate probabilities and entropy
            original_probs = F.softmax(original_logits, dim=-1)
            modified_probs = F.softmax(modified_logits, dim=-1)

            # Calculate entropy: -sum(p * log(p))
            original_entropy = -torch.sum(original_probs * torch.log(original_probs + 1e-10), dim=-1)
            modified_entropy = -torch.sum(modified_probs * torch.log(modified_probs + 1e-10), dim=-1)

            # Get predicted tokens
            original_tokens = torch.argmax(original_logits, dim=-1)
            modified_tokens = torch.argmax(modified_logits, dim=-1)

            # Debug: Print tensor shapes

            # Handle different tensor shapes
            if len(original_tokens.shape) == 1:
                # If 1D tensor, treat as single batch
                batch_size = 1
                seq_len = original_tokens.shape[0]
                original_tokens = original_tokens.unsqueeze(0)
                modified_tokens = modified_tokens.unsqueeze(0)
                original_entropy = original_entropy.unsqueeze(0)
                modified_entropy = modified_entropy.unsqueeze(0)
            elif len(original_tokens.shape) == 2:
                batch_size, seq_len = original_tokens.shape
            else:
                # Handle higher dimensions by flattening
                original_shape = original_tokens.shape
                original_tokens = original_tokens.view(-1, original_shape[-1])
                modified_tokens = modified_tokens.view(-1, original_shape[-1])
                original_entropy = original_entropy.view(-1, original_shape[-1])
                modified_entropy = modified_entropy.view(-1, original_shape[-1])
                batch_size, seq_len = original_tokens.shape

            # Prepare response data list
            response_data = []

            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    # Create record for each response token
                    record = {
                        "batch_idx": batch_idx,
                        "seq_idx": seq_idx,
                        "original_entropy": original_entropy[batch_idx, seq_idx].item(),
                        "modified_entropy": modified_entropy[batch_idx, seq_idx].item(),
                        "entropy_diff": (
                                modified_entropy[batch_idx, seq_idx] - original_entropy[batch_idx, seq_idx]).item(),
                        "original_token": original_tokens[batch_idx, seq_idx].item(),
                        "modified_token": modified_tokens[batch_idx, seq_idx].item(),
                        "original_token_decoded": _safe_decode_token(original_tokens[batch_idx, seq_idx].item()),
                        "modified_token_decoded": _safe_decode_token(modified_tokens[batch_idx, seq_idx].item()),
                    }
                    response_data.append(record)

            # Read existing data if file exists
            existing_data = []
            if os.path.exists(response_entropy_file):
                try:
                    with open(response_entropy_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []

            # Extend existing data with new response data
            existing_data.extend(response_data)

            # Write updated data back to file
            with open(response_entropy_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        # Log error but don't interrupt the forward pass
        import traceback
        print(f"Error in response entropy analysis: {e}")
        print(f"Traceback: {traceback.format_exc()}")


def apply_entropy_control(model, logits, past_key_values, input_ids, logits_to_keep=0, stage="generation"):
    """Apply entropy-based early stopping logic， Second Key module, 就是Check 5之后的内容会直接跳到这里"""

    # Add entropy-based early stopping logic
    entropy_control_enabled = os.environ.get("entropy_control", "False") == "True"
    if entropy_control_enabled and logits.shape[
        1] > 0 and stage == "generation":
        entropy_threshold = float(os.environ.get("entropy_threshold", "3.0"))
        long_entropy_threshold = float(os.environ.get("long_entropy_threshold", "3.0"))
        entropy, _ = cal_uncertainty(logits, uncertain_name='entropy')

        if os.environ.get("adaptive_entropy", "False") == "True":
            adaptive_entropy_N = int(os.environ.get("adaptive_entropy_N", "20"))  # Windows长度，这个要根据query的长度分布来看下。
            adaptive_entropy_K = float(os.environ.get("adaptive_entropy_K", "2"))  # 强化的倍数

            long_adaptive_entropy_N = int(
                os.environ.get("long_adaptive_entropy_N", "35"))
            long_adaptive_entropy_K = float(os.environ.get("long_adaptive_entropy_K", "2"))

            current_len = len(model.entropy_history) + 1

            if current_len > long_adaptive_entropy_N:
                window = torch.tensor(model.entropy_history[-adaptive_entropy_N:], device=entropy.device)
                long_window = torch.tensor(model.entropy_history[-long_adaptive_entropy_N:], device=entropy.device)

                minimal_std = float(os.environ.get("minimal_std", "0.5"))
                minimal_threshold = float(os.environ.get("minimal_threshold", "1.8"))

                mean_history = torch.mean(window)
                std_history = max(torch.std(window), minimal_std)

                long_mean_history = torch.mean(long_window)
                long_std_history = max(torch.std(long_window), minimal_std)

                # print("=====Check 7.1", mean_history, std_history)

                entropy_threshold = mean_history + adaptive_entropy_K * std_history
                entropy_threshold = entropy_threshold.item()
                entropy_threshold = max(entropy_threshold, minimal_threshold)
                long_entropy_threshold = long_mean_history + long_adaptive_entropy_K * long_std_history
                long_entropy_threshold = long_entropy_threshold.item()
                long_entropy_threshold = max(long_entropy_threshold, minimal_threshold)  # Ensure non-negative threshold

            model.entropy_history.append(entropy.item())  # [[batch_size]]
            model.long_entropy_history.append(entropy.item())  # [[batch_size]]

        # Check if entropy exceeds threshold
        high_entropy_mask = entropy > entropy_threshold
        long_high_entropy_mask = entropy > long_entropy_threshold
        high_entropy_mask = high_entropy_mask & long_high_entropy_mask

        if high_entropy_mask.any():

            if response_entropy_file := os.environ.get("response_entropy_file", ""):
                _record_high_entropy_token(model, logits, logits_to_keep,
                                           response_entropy_file)
            # Mark that high entropy was detected
            model.high_entropy_detected = True
            # Get the current sequence length for position tracking
            if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
                current_length = past_key_values.get_seq_length() + logits.shape[1]
            elif input_ids is not None:
                current_length = input_ids.shape[1]
            else:
                current_length = 0
            model.high_entropy_position = current_length

            # Force EOS token for high entropy samples
            eos_token_id = getattr(model.config, 'eos_token_id', 2)  # Default to 2 if not specified

            # Create new logits with EOS token as the highest probability
            modified_logits = logits.clone()

            dtype = modified_logits.dtype
            large_value = 1e4
            if os.environ.get("log_entropy_control", "False") == "True":
                print(f"large_value: {large_value:.4f} for dtype {dtype}")

            for batch_idx in range(logits.shape[0]):
                if high_entropy_mask[
                    batch_idx]:
                    modified_logits[batch_idx, -1, :] = -large_value
                    modified_logits[batch_idx, -1, eos_token_id] = large_value

            logits = modified_logits

            # Log entropy detection
            if os.environ.get("log_entropy_control", "False") == "True":
                for batch_idx in range(entropy.shape[0]):
                    if high_entropy_mask[batch_idx]:
                        print(
                            f"High entropy detected: {entropy[batch_idx].item():.4f} > {entropy_threshold} at position {current_length}")

    return logits


def _reset_entropy_detection(model):
    """Reset entropy detection state for new generation"""
    model.high_entropy_detected = False
    model.high_entropy_position = None


def _reset_model_parameters(model):
    """Reset model parameters"""
    model.delta = None
    model.entropy_threshold = None
    model.entropy_history = []
    model.long_entropy_history = []


def _reset_offline_parameters(model):
    """Set model parameters for offline use"""
    model.delta_normal = None
    model.delta_high = None
    model.delta_stru = None
    model.delta_query = None


def _reset_offline_parameters_method(self):
    """Method wrapper for _set_offline_parameters"""
    return _reset_offline_parameters(self)


def _reset_entropy_detection_method(self):
    """Method wrapper for reset_entropy_detection"""
    return _reset_entropy_detection(self)


def _reset_model_parameters_method(self):
    """Method wrapper for reset_model_parameters"""
    return _reset_model_parameters(self)


def _safe_decode_token_method(self, token_id):
    """Method wrapper for _safe_decode_token"""
    return _safe_decode_token(token_id)


def _safe_decode_sequence_method(self, token_ids):
    """Method wrapper for _safe_decode_sequence"""
    return _safe_decode_sequence(token_ids)


def _safe_decode_token(token_id):
    """Safely decode a token ID to text, handling potential errors"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(os.environ.get("tokenizer_path"))
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        decoded = decoded.strip()
        if not decoded:  # If empty after stripping
            decoded = tokenizer.convert_ids_to_tokens([token_id])[0]
        return decoded
    except Exception as e:
        return f"<decode_error_{token_id}>"


def _safe_decode_sequence(token_ids):
    """Safely decode a sequence of token IDs to text"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(os.environ.get("tokenizer_path"))
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        return decoded
    except Exception as e:
        return f"<decode_error: {e}>"


def _add_dsi_methods(model):
    """Add all the DSI-specific methods to the model instance"""
    model.reset_entropy_detection = functools.partial(_reset_entropy_detection, model)
    model.reset_model_parameters = functools.partial(_reset_model_parameters, model)
    model._safe_decode_token = _safe_decode_token
    model._safe_decode_sequence = _safe_decode_sequence
    model._record_response_entropy = functools.partial(_record_response_entropy, model)

    model.reset_offline_parameters = functools.partial(_reset_offline_parameters, model)
