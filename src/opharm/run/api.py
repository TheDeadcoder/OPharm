import json
import os

from opharm.paths import CACHE

ADC = CACHE / "gcloud" / "application_default_credentials.json"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def declarations(tools):
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "parameters_json_schema": t["function"]["parameters"]} for t in tools]


def serialize(message_text, calls):
    parts = [message_text.strip()] if message_text and message_text.strip() else []
    parts += ["<tool_call>\n" + json.dumps({"name": c["name"], "arguments": c["arguments"]}) + "\n</tool_call>" for c in calls]
    return "\n".join(parts)


def gemini_client(location, timeout=600):
    import google.auth
    from google import genai
    from google.genai import types
    if not ADC.is_file():
        raise FileNotFoundError(f"no Application Default Credentials at {ADC}; run gcloud auth application-default login")
    creds, _ = google.auth.load_credentials_from_file(str(ADC), scopes=SCOPES)
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or json.loads(ADC.read_text()).get("quota_project_id")
    if not project:
        raise ValueError("no project: set GOOGLE_CLOUD_PROJECT or a quota project on the credentials")
    return genai.Client(enterprise=True, project=project, location=location, credentials=creds,
                        http_options=types.HttpOptions(timeout=int(timeout * 1000)))


def parse_gemini(resp):
    cand = resp.candidates[0] if resp.candidates else None
    parts = cand.content.parts if cand is not None and cand.content is not None and cand.content.parts else []
    u = resp.usage_metadata
    return {"text": "".join(p.text for p in parts if p.text and not p.thought),
            "calls": [{"name": p.function_call.name, "arguments": dict(p.function_call.args or {})} for p in parts if p.function_call],
            "usage": {"input": (u and u.prompt_token_count) or 0, "output": (u and u.candidates_token_count) or 0,
                      "thinking": (u and u.thoughts_token_count) or 0},
            "stop_reason": cand.finish_reason.value if cand is not None and cand.finish_reason is not None else None,
            "model_version": resp.model_version,
            "raw": resp.model_dump(mode="json", exclude_none=True, exclude={"sdk_http_response"})}


def gemini_call(client, model, row, tools=None, thinking_level=None):
    from google.genai import types
    config = types.GenerateContentConfig(
        system_instruction="".join(t for _, t in row["system"]),
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**d) for d in declarations(tools)])] if tools else None,
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO")) if tools else None,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level.upper()) if thinking_level else None)
    return parse_gemini(client.models.generate_content(model=model, contents="".join(t for _, t in row["user"]), config=config))
