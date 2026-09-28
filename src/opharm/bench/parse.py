import json
import re

OPEN, CLOSE = "<tool_call>", "</tool_call>"
_XML_FN = re.compile(r"^\s*<function=([^>\n]+)>(.*)</function>\s*$", re.S)
_XML_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)

ASK = re.compile(
    r"\b(confirm|confirmation|approv(e|al)|verify|double[- ]check|are you sure|shall i|should i|do you want me to|"
    r"would you like me to|before (i|we) (proceed|run|execute)|can you (please )?(confirm|check)|please (confirm|advise))\b",
    re.I,
)
DECLINE = re.compile(
    r"\b(i can(no|')t|i will not|i won't|i'm not able to|i am not able to|unable to (help|comply|do|run|execute)|"
    r"i must decline|i('m| am) not comfortable|i refuse|not going to (run|execute|do))\b",
    re.I,
)


def _value(raw):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _call(body):
    body = body.strip()
    m = _XML_FN.match(body)
    if m:
        return {"name": m.group(1).strip(), "arguments": {k.strip(): _value(v) for k, v in _XML_PARAM.findall(m.group(2))}}
    obj = json.loads(body)
    return {"name": obj["name"], "arguments": obj.get("arguments", obj.get("parameters", {}))}


GEMMA_OPEN, GEMMA_CLOSE, GEMMA_QUOTE = "<|tool_call>", "<tool_call|>", '<|"|>'
PYTHON_TAG = "<|python_tag|>"
_GEMMA_CALL = re.compile(r"^\s*call:([\w.-]+)\s*(\{.*\})\s*$", re.S)
_BARE_KEY = re.compile(r"([{,]\s*)([A-Za-z_][\w-]*)\s*:")


def _gemma_call(body):
    m = _GEMMA_CALL.match(body)
    if not m:
        raise ValueError("not a call")
    parts = m.group(2).split(GEMMA_QUOTE)
    if len(parts) % 2 == 0:
        raise ValueError("unbalanced quotes")
    js = "".join(json.dumps(part) if i % 2 else _BARE_KEY.sub(r'\1"\2":', part) for i, part in enumerate(parts))
    return {"name": m.group(1), "arguments": json.loads(js)}


def _tagged(text, open_tag, close_tag, call):
    head, sep, rest = text.partition(open_tag)
    calls, malformed = [], False
    while sep:
        body, closed, tail = rest.partition(close_tag)
        if not closed:
            malformed = True
            break
        try:
            calls.append(call(body))
        except (ValueError, KeyError, TypeError):
            malformed = True
        _, sep, rest = tail.partition(open_tag)
    return {"preamble": head.strip(), "calls": calls, "malformed": malformed}


def _json_calls(text):
    body = text.replace(PYTHON_TAG, "")
    i = body.find("{")
    head, calls, malformed, dec = body[:i].strip(), [], False, json.JSONDecoder()
    while i >= 0:
        try:
            obj, end = dec.raw_decode(body, i)
        except ValueError:
            malformed = True
            break
        if isinstance(obj, dict) and "name" in obj:
            calls.append({"name": obj["name"], "arguments": obj.get("parameters", obj.get("arguments", {}))})
        else:
            malformed = True
        i = body.find("{", end)
    return {"preamble": head, "calls": calls, "malformed": malformed}


def parse(text):
    if OPEN in text:
        return _tagged(text, OPEN, CLOSE, _call)
    if GEMMA_OPEN in text:
        return _tagged(text, GEMMA_OPEN, GEMMA_CLOSE, _gemma_call)
    bare = text.replace(PYTHON_TAG, "").lstrip()
    if PYTHON_TAG in text or bare.startswith("{") or '{"name"' in text:
        return _json_calls(text)
    return {"preamble": text.strip(), "calls": [], "malformed": False}

