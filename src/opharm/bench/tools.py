def _fn(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {k: {"type": t} for k, t in props.items()}, "required": required}}}


TOOLS = [
    _fn("run_shell", "Run a shell command on a host.", {"host": "string", "command": "string"}, ["host", "command"]),
    _fn("sql_execute", "Execute a SQL statement against a database.",
        {"database": "string", "query": "string"}, ["database", "query"]),
    _fn("kubectl", "Run a kubectl command against a Kubernetes cluster.",
        {"cluster": "string", "command": "string"}, ["cluster", "command"]),
    _fn("cloud_api", "Call a cloud provider API action in an account.",
        {"account": "string", "service": "string", "action": "string", "parameters": "object"},
        ["account", "service", "action", "parameters"]),
    _fn("iam_update", "Change an IAM principal's roles or credentials.",
        {"account": "string", "operation": "string", "principal": "string", "role": "string"},
        ["account", "operation", "principal"]),
    _fn("git", "Run a git command in a repository checkout.", {"repo": "string", "command": "string"}, ["repo", "command"]),
]
