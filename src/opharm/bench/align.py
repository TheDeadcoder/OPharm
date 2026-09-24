def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def same_length(toks, a, b):
    return all(n_tokens(t, a) == n_tokens(t, b) for t in toks)


def aligned_pairs(toks, left, right, context="{}"):
    return [(a, b) for a in left for b in right if same_length(toks, context.format(a), context.format(b))]


def pair_exclusion(ra, rb, span):
    if len(ra.ids) != len(rb.ids):
        return "length"
    if ra.spans.get(span) != rb.spans.get(span):
        return "span_position"
    inside = {i for s, e in ra.spans.get(span, []) for i in range(s, e)}
    if any(x != y and i not in inside for i, (x, y) in enumerate(zip(ra.ids, rb.ids))):
        return "outside_span"
    if ra.t_inst != rb.t_inst:
        return "t_inst"
    return None
