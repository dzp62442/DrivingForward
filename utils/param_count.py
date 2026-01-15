def _count_params_from_list(params, trainable_only=None):
    if trainable_only is None:
        return sum(p.numel() for p in params)
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params if not p.requires_grad)


def _format_param_count(count):
    if count >= 1_000_000_000:
        value = count / 1_000_000_000
        unit = "B"
    elif count >= 1_000_000:
        value = count / 1_000_000
        unit = "M"
    elif count >= 1_000:
        value = count / 1_000
        unit = "K"
    else:
        return str(count)
    if value >= 100:
        text = f"{value:.0f} {unit}"
    elif value >= 10:
        text = f"{value:.1f} {unit}"
    else:
        text = f"{value:.2f} {unit}"
    return text


def build_param_table(model):
    """自动统计一级模块与 others 的参数量表。"""
    rows = []
    module_params_ids = set()
    children = list(model.named_children())
    for name, module in children:
        params = list(module.parameters())
        for p in params:
            module_params_ids.add(id(p))
        total = _count_params_from_list(params, trainable_only=None)
        trainable = _count_params_from_list(params, trainable_only=True)
        frozen = total - trainable
        rows.append(
            (
                name,
                module.__class__.__name__,
                _format_param_count(trainable),
                _format_param_count(frozen),
                _format_param_count(total),
                "train" if module.training else "eval",
            )
        )

    all_params = list(model.parameters())
    other_params = [p for p in all_params if id(p) not in module_params_ids]
    other_total = _count_params_from_list(other_params, trainable_only=None)
    other_trainable = _count_params_from_list(other_params, trainable_only=True)
    other_frozen = other_total - other_trainable
    rows.append(
        (
            "others",
            "params",
            _format_param_count(other_trainable),
            _format_param_count(other_frozen),
            _format_param_count(other_total),
            "n/a",
        )
    )

    name_w = max([len("Name")] + [len(r[0]) for r in rows]) if rows else len("Name")
    type_w = max([len("Type")] + [len(r[1]) for r in rows]) if rows else len("Type")
    train_w = max([len("Trainable")] + [len(r[2]) for r in rows]) if rows else len("Trainable")
    frozen_w = max([len("Non-trainable")] + [len(r[3]) for r in rows]) if rows else len("Non-trainable")
    total_w = max([len("Total")] + [len(r[4]) for r in rows]) if rows else len("Total")
    mode_w = max([len("Mode")] + [len(r[5]) for r in rows]) if rows else len("Mode")
    idx_w = max(1, len(str(len(rows) - 1))) if rows else 1
    header = (
        f"  | {'Name':<{name_w}} | {'Type':<{type_w}} | {'Trainable':<{train_w}} | "
        f"{'Non-trainable':<{frozen_w}} | {'Total':<{total_w}} | {'Mode':<{mode_w}}"
    )
    row_lines = [
        f"{idx:{idx_w}d} | {name:<{name_w}} | {type_name:<{type_w}} | {trainable:<{train_w}} | "
        f"{frozen:<{frozen_w}} | {total:<{total_w}} | {mode:<{mode_w}}"
        for idx, (name, type_name, trainable, frozen, total, mode) in enumerate(rows)
    ]
    sep_len = max([len(header)] + [len(line) for line in row_lines]) if row_lines else len(header)
    sep = "-" * sep_len

    total_all = _count_params_from_list(all_params, trainable_only=None)
    total_trainable = _count_params_from_list(all_params, trainable_only=True)
    total_frozen = total_all - total_trainable
    total_lines = [
        f"{_format_param_count(total_trainable):<{total_w}}     Trainable params",
        f"{_format_param_count(total_frozen):<{total_w}}     Non-trainable params",
        f"{_format_param_count(total_all):<{total_w}}     Total params",
    ]
    return [header, sep] + row_lines + [sep] + total_lines
