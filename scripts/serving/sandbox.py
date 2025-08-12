from fastapi import FastAPI
import contextlib
import io
import ast
from typing import Any, Dict, List, Union
from pydantic import BaseModel
from argparse import ArgumentParser
import sympy as sp
import numpy as np
import pandas as pd
import json
import asyncio
import os
import tempfile

app = FastAPI()


def serialize_object(obj: Any) -> Union[str, Dict, List, float, int, bool, None]:
    if obj is None:
        return None
    elif isinstance(obj, (int, float, bool, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [serialize_object(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_object(value) for key, value in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, sp.Expr):
        return sp.printing.sstr(obj)
    elif hasattr(obj, '__dict__'):
        return serialize_object(obj.__dict__)
    else:
        try:
            return json.dumps(obj)
        except (TypeError, OverflowError):
            return str(obj)


def get_last_assigned_variable_name(code: str):
    try:
        tree = ast.parse(code)
        for stmt in reversed(tree.body):
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets[0]
                if isinstance(targets, ast.Name):
                    return [targets.id]
                elif isinstance(targets, ast.Tuple):
                    return [elt.id for elt in targets.elts if isinstance(elt, ast.Name)]
        return None
    except Exception:
        return None


class CodeRequest(BaseModel):
    code: str
    language: str = "python"
    timeout: int = 10


@app.post("/execute")
async def execute_code(request: CodeRequest) -> Dict:
    print("-" * 30)
    print(request.code)
    print("-" * 30)

    if request.language != "python":
        return {"output": "", "result": None, "error": "Unsupported language"}

    import inspect
    serialize_src = inspect.getsource(serialize_object)
    get_last_src = inspect.getsource(get_last_assigned_variable_name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as runner_file:
        runner_path = runner_file.name
        runner_script = f"""
import sys, json, io, ast
import sympy as sp
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
from contextlib import redirect_stdout

{serialize_src}

{get_last_src}

code = sys.stdin.read()

output = io.StringIO()
result = None
error = None
last_vars = None

try:
    with redirect_stdout(output):
        exec_globals = {{"__builtins__": __builtins__}}
        exec(code, exec_globals)
        last_vars = get_last_assigned_variable_name(code)
        if last_vars:
            result_dict = {{}}
            for name in last_vars:
                if name in exec_globals:
                    result_dict[name] = exec_globals[name]
                else:
                    result_dict[name] = None
            result = serialize_object(result_dict)     
            # result = exec_globals.get(last_vars)
            # result = serialize_object(result)
except Exception as e:
    error = str(e)

print(json.dumps({{
    "output": output.getvalue(),
    "result": result,
    "error": error
}}))
"""
        runner_file.write(runner_script)

    cmd = ['python', runner_path]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    result_data = {}
    try:

        stdout, stderr = await asyncio.wait_for(
            process.communicate(input=request.code.encode('utf-8')),
            timeout=request.timeout
        )
    except asyncio.TimeoutError:

        process.kill()
        await process.wait()
        result_data = {
            "output": "",
            "result": None,
            "error": f"Execution timed out after {request.timeout} seconds"
        }
    else:
        stdout_str = stdout.decode('utf-8', errors='replace')
        stderr_str = stderr.decode('utf-8', errors='replace')

        if process.returncode != 0:
            result_data = {
                "output": stdout_str,
                "result": None,
                "error": stderr_str or f"Process exited with code {process.returncode}"
            }
        else:
            try:
                result_data = json.loads(stdout_str)
            except json.JSONDecodeError:
                result_data = {
                    "output": stdout_str,
                    "result": None,
                    "error": "Failed to parse result from subprocess"
                }
    finally:
        try:
            os.unlink(runner_path)
        except:
            pass

    def truncate(value, length=50):
        if isinstance(value, str):
            return value[:length]
        elif isinstance(value, int):
            return str(value)[:length]
        elif isinstance(value, list):
            return value[:length]
        elif isinstance(value, dict):
            value_dict = {}
            for k, v in value.items():
                value_dict[k] = truncate(v)
            return value_dict

        return value


    return {
        "output": truncate(result_data.get("output", ""), 50),
        "result": truncate(result_data.get("result"), 50),
        "error": truncate(result_data.get("error"), 50)
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=81)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=args.port)