"""
Core evaluation utilities for language models.
Handles multiple-choice, schema, and language modeling tasks.
"""
import random
import torch
import torch.distributed as dist
from jinja2 import Template
import weave

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple-choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.question }}
{% for choice in example.choices %}{{ loop.index0|string + ". " + choice }}
{% endfor %}Answer:{{ continuation_delimiter}}{{ example.gold|string }}

{% endfor -%}
{{ item.question }}
{% for choice in item.choices %}{{ loop.index0|string + ". " + choice }}
{% endfor %}Answer:{{ continuation_delimiter}}{{ choice_idx|string }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice_idx=i, **context)
               for i in range(len(item['choices']))]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a language modeling question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context }}{{ continuation_delimiter}}""".strip()
    template_without_continuation = Template(template_str)
    template_with_continuation = Template(template_str + "{{ item.continuation }}")
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompt_without = template_without_continuation.render(**context)
    prompt_with = template_with_continuation.render(**context)
    return [prompt_without, prompt_with]


def find_common_length(sequences, direction='left'):
    """Find common prefix (direction='left') or suffix (direction='right') length"""
    if not sequences or len(sequences) < 2:
        return 0
    if direction == 'left':
        common_len = 0
        for i in range(min(len(s) for s in sequences)):
            if all(s[i] == sequences[0][i] for s in sequences):
                common_len += 1
            else:
                break
        return common_len
    else:  # direction == 'right'
        common_len = 0
        for i in range(1, min(len(s) for s in sequences) + 1):
            if all(s[-i] == sequences[0][-i] for s in sequences):
                common_len += 1
            else:
                break
        return common_len


def batch_sequences_mc(tokenizer, prompts):
    # In multiple-choice tasks, all prompts share a common prefix (the question)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation (the answer choice)
    prefix_length = find_common_length(tokens, direction='left')
    start_indices = [prefix_length] * len(tokens)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each context
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids):
    """
    Forward the model and return losses and predictions.
    Losses are of shape (B, T) and predictions are of shape (B, T) where B is batch size and T is sequence length.
    Note that the losses and predictions are shifted: losses[i] corresponds to the loss for predicting input_ids[i+1].
    """
    # Forward pass
    logits = model(input_ids)  # (B, T, V)
    # Get predictions
    predictions = logits.argmax(dim=-1)  # (B, T)
    # Calculate per-token loss
    # shift logits and targets so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shift_labels = input_ids[:, 1:].contiguous()  # (B, T-1)
    # compute cross entropy loss for each token
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    losses = losses.view(shift_labels.size())  # (B, T-1)

    return losses, predictions


def stack_sequences(sequences, pad_token_id):
    """Stack variable-length sequences into a padded tensor"""
    max_length = max(len(s) for s in sequences)
    padded = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded


@weave.op()
@torch.no_grad()
def evaluate_language_modeling_example(item, model, tokenizer, device, task_name, continuation_delimiter, num_fewshot, fewshot_examples):
    """Evaluate a single language modeling example"""
    # Render prompts and batch sequences
    prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
    tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    
    # Truncate if needed
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Decode input for logging
    model_input = tokenizer.decode(tokens[0][:start_idxs[0]]) if tokens else ""

    # Stack and forward
    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    losses, predictions = forward_model(model, input_ids)

    # Evaluate
    si, ei = start_idxs[0], end_idxs[0]
    predicted_tokens = predictions[0, si-1:ei-1]
    actual_tokens = input_ids[0, si:ei]
    is_correct = torch.all(predicted_tokens == actual_tokens).item()
    
    predicted_text = tokenizer.decode(predicted_tokens.cpu().tolist())
    actual_text = tokenizer.decode(actual_tokens.cpu().tolist())
    
    return {
        "is_correct": is_correct,
        "task_name": task_name,
        "task_type": "language_modeling",
        "model_input": model_input,
        "predicted_continuation": predicted_text,
        "actual_continuation": actual_text,
        "num_fewshot": num_fewshot,
    }


@weave.op()
@torch.no_grad()
def evaluate_multiple_choice_example(item, model, tokenizer, device, task_name, continuation_delimiter, num_fewshot, fewshot_examples):
    """Evaluate a single multiple-choice example"""
    # Render prompts and batch sequences
    prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
    tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    
    # Truncate if needed
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack and forward
    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    losses, predictions = forward_model(model, input_ids)

    # Find option with lowest loss
    mean_losses = [losses[i, si-1:ei-1].mean().item()
                    for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
    pred_idx = mean_losses.index(min(mean_losses))
    is_correct = pred_idx == item['gold']
    
    predicted_choice = item['choices'][pred_idx] if pred_idx < len(item['choices']) else "unknown"
    correct_choice = item['choices'][item['gold']] if item['gold'] < len(item['choices']) else "unknown"
    
    return {
        "is_correct": is_correct,
        "task_name": task_name,
        "task_type": "multiple_choice",
        "question": item.get('question', ''),
        "choices": item.get('choices', []),
        "predicted_choice": predicted_choice,
        "correct_choice": correct_choice,
        "num_fewshot": num_fewshot,
    }


@weave.op()
@torch.no_grad()
def evaluate_schema_example(item, model, tokenizer, device, task_name, continuation_delimiter, num_fewshot, fewshot_examples):
    """Evaluate a single schema example"""
    # Render prompts and batch sequences
    prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
    tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    
    # Truncate if needed
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack and forward
    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    losses, predictions = forward_model(model, input_ids)

    # Find option with lowest loss
    mean_losses = [losses[i, si-1:ei-1].mean().item()
                    for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
    pred_idx = mean_losses.index(min(mean_losses))
    is_correct = pred_idx == item['gold']
    
    # Show full prompts for clarity
    continuation = item.get('continuation', '')
    context_option_one = item['context_options'][0] + continuation_delimiter + continuation if len(item['context_options']) > 0 else ""
    context_option_two = item['context_options'][1] + continuation_delimiter + continuation if len(item['context_options']) > 1 else ""
    
    return {
        "is_correct": is_correct,
        "task_name": task_name,
        "task_type": "schema",
        "context_option_one": context_option_one,
        "context_option_two": context_option_two,
        "predicted_context_idx": pred_idx,
        "correct_context_idx": item['gold'],
        "num_fewshot": num_fewshot,
    }


@torch.no_grad()
def evaluate_example(item, model, tokenizer, device, task_meta, fewshot_examples=None):
    """Evaluate a single example - dispatches to task-specific traced functions"""
    task_name = task_meta.get('task_name', 'unknown')
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']
    fewshot_examples = fewshot_examples or []

    # Dispatch to task-specific function (each is separately traced by Weave)
    if task_type == 'language_modeling':
        return evaluate_language_modeling_example(
            item, model, tokenizer, device, task_name, 
            continuation_delimiter, num_fewshot, fewshot_examples
        )
    elif task_type == 'multiple_choice':
        return evaluate_multiple_choice_example(
            item, model, tokenizer, device, task_name,
            continuation_delimiter, num_fewshot, fewshot_examples
        )
    elif task_type == 'schema':
        return evaluate_schema_example(
            item, model, tokenizer, device, task_name,
            continuation_delimiter, num_fewshot, fewshot_examples
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    num_fewshot = task_meta['num_fewshot']
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    # stride the examples to each rank
    for idx in range(rank, len(data), world_size):
        item = data[idx]
        # Sample few-shot examples (excluding current item)
        fewshot_examples = []
        if num_fewshot > 0:
            rng = random.Random(1234 + idx)
            # sample without replacement, excluding the current item
            indices = list(range(len(data)))
            indices.pop(idx)
            sampled_indices = rng.sample(indices, min(num_fewshot, len(indices)))
            fewshot_examples = [data[i] for i in sampled_indices]
        # Evaluate the example
        result = evaluate_example(item, model, tokenizer, device, task_meta, fewshot_examples)
        correct[idx] = result["is_correct"]
    # synchronize results across ranks
    if dist.is_initialized():
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # calculate accuracy
    accuracy = correct.sum().item() / len(data)
    return {"accuracy": accuracy}
