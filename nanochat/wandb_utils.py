"""Small W&B compatibility helpers shared by training and evaluation scripts."""

import wandb


def make_eval_table(input_columns, output_columns, score_columns, data):
    """Build an EvalTable when supported, with a non-fatal legacy fallback."""
    eval_table_cls = getattr(wandb, "EvalTable", None)
    if eval_table_cls is not None:
        return eval_table_cls(
            input_columns=input_columns,
            output_columns=output_columns,
            score_columns=score_columns,
            data=data,
        )
    print("Warning: this wandb version has no EvalTable; logging a regular Table instead")
    return wandb.Table(
        columns=[*input_columns, *output_columns, *score_columns],
        data=data,
    )