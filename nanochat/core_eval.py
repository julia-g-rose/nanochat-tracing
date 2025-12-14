"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
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
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
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
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
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
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


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

    # Extract the question part (common prefix before choices) for logging
    question_text = tokenizer.decode(tokens[0][:start_idxs[0]]) if tokens and start_idxs else item.get('question', '')
    # Raw question from the dataset (if provided)
    question_raw = item.get('question', '')
    # Rendered prompt (text the model actually saw for choice 0)
    prompt_rendered = prompts[0] if prompts else question_text

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
        "question": question_raw or question_text,
        "question_rendered": question_text,
        "prompt_rendered": prompt_rendered,
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
    
    # Extract actual context options and continuation from tokens
    # In schema tasks: start_idx marks where context ends and continuation begins
    context_options = [tokenizer.decode(tokens[i][:start_idxs[i]]) for i in range(len(tokens))]
    continuation = tokenizer.decode(tokens[0][start_idxs[0]:end_idxs[0]]) if tokens else item.get('continuation', '')
    
    return {
        "is_correct": is_correct,
        "task_name": task_name,
        "task_type": "schema",
        "context_options": context_options,
        "continuation": continuation,
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
            available_indices = [i for i in range(len(data)) if i != idx]
            fewshot_indices = rng.sample(available_indices, num_fewshot)
            fewshot_examples = [data[i] for i in fewshot_indices]
        # Evaluate the example
        result = evaluate_example(item, model, tokenizer, device, task_meta, fewshot_examples)
        correct[idx] = result["is_correct"]
    # synchronize results across ranks
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # calculate accuracy
    mean_correct = correct.mean().item()
    return {"accuracy": mean_correct}
