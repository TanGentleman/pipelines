"""
title: Llama C++ Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for generating responses using the Llama C++ library.
requirements: llama-cpp-python
"""

from typing import Callable, List, Optional, Union, Generator, Iterator

from pydantic import BaseModel

# from constants import EXAMPLE_SCHEMA_STRING, AudioFeature, Car, CarAndOwner
# from schemas import OpenAIChatMessage
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from os.path import expanduser
MODELS_DIR = expanduser("~/.models/lm-studio/")
OPENCHAT_MODEL_PATH = "bartowski/openchat-3.6-8b-20240522-GGUF/openchat-3.6-8b-20240522-Q6_K.gguf"
GEMMA_MODEL_PATH = "lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q8_0.gguf"
SMOL_GEMMA_PATH = "QuantFactory/gemma-2-2b-it-abliterated-GGUF/gemma-2-2b-it-abliterated.Q4_0.gguf"
MODEL_PATH = MODELS_DIR + SMOL_GEMMA_PATH

# RESPONSE_SCHEMA_STRING = (json.dumps(Summary.model_json_schema(), indent=2))
# WEB_SCHEMA = r'''
# space ::= " "?
# string ::=  "\"" (
#         [^"\\] |
#         "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
#       )* "\"" space 
# Stage ::= "\"first\"" | "\"second\""
# boolean ::= ("true" | "false") space
# root ::= "{" space "\"Assistant\"" space ":" space string "," space "\"Stage\"" space ":" space Stage "," space "\"Statement\"" space ":" space string "," space "\"Task Finished\"" space ":" space boolean "}" space
# '''
# WEB_SCHEMA = r'''
# space ::= " "?
# string ::=  "\"" (
#         [^"\\] |
#         "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
#       )* "\"" space 
# Stage ::= "\"first\"" | "\"second\""
# boolean ::= ("true" | "false") space
# object ::= "{" space "\"Assistant\"" space ":" space string "," space "\"Stage\"" space ":" space Stage "," space "\"Statement\"" space ":" space string "," space "\"Task Finished\"" space ":" space boolean "}" space
# array ::= "[" space object ("," space object)+ "]" space
# root ::= array
# '''
# # Grammar that only enforces that the first character is {:
# SIMPLE_SCHEMA = r'''
# space ::= " "?
# root ::= "{" space
# '''

def calculate_token_rate(time_ms, num_tokens):
    """
    Calculate token rate in tokens per second.

    Args:
        time_ms (float): Time in milliseconds.
        num_tokens (int): Number of tokens.

    Returns:
        float: Token rate in tokens per second.
    """
    if not time_ms:
        print("ERROR:No time!")
        return 0
    if not num_tokens:
        print("ERROR:No tokens!")
        return 0
    return num_tokens / (time_ms / 1000)

def calculate_time_per_token(time_ms, num_tokens):
    """
    Calculate time per token in milliseconds.

    Args:
        time_ms (float): Time in milliseconds.
        num_tokens (int): Number of tokens.

    Returns:
        float: Time per token in milliseconds.
    """
    if not num_tokens:
        print("ERROR:No tokens!")
        return 0
    if not time_ms:
        print("ERROR:No time!")
        return 0
    return time_ms / num_tokens

def format_timing(label, time_ms, num_runs=None, num_tokens=None):
    """
    Format a timing line.

    Args:
        label (str): Timing label.
        time_ms (float): Time in milliseconds.
        num_runs (int, optional): Number of runs. Defaults to None.
        num_tokens (int, optional): Number of tokens. Defaults to None.

    Returns:
        str: Formatted timing line.
    """
    if num_runs is not None:
        time_per_token = calculate_time_per_token(time_ms, num_runs)
        token_rate = calculate_token_rate(time_ms, num_runs)
        return f"{label} = {time_ms:.2f} ms / {num_runs} runs   ( {time_per_token:.2f} ms per token, {token_rate:.2f} tokens per second)"
    elif num_tokens is not None:
        time_per_token = calculate_time_per_token(time_ms, num_tokens)
        token_rate = calculate_token_rate(time_ms, num_tokens)
        return f"{label} = {time_ms:.2f} ms / {num_tokens} tokens ( {time_per_token:.2f} ms per token, {token_rate:.2f} tokens per second)"
    else:
        return f"{label} = {time_ms:.2f} ms"

def unpack_llama_timings(timings):
    """
    Unpack llama timings into a dictionary.

    Args:
        timings (llama_timings): Llama timings object.

    Returns:
        dict: Unpacked llama timings dictionary.
    """
    return {
        "load_time": timings.t_load_ms,
        "sample_time": timings.t_sample_ms,
        "sample_runs": timings.n_sample,
        "prompt_eval_time": timings.t_p_eval_ms,
        "prompt_eval_tokens": timings.n_p_eval,
        "eval_time": timings.t_eval_ms,
        "eval_runs": timings.n_eval,
        "total_time": timings.t_end_ms - timings.t_start_ms,
        "total_tokens": timings.n_p_eval + timings.n_eval,
    }

def get_timing_dict(timings):
    unpacked_timings = unpack_llama_timings(timings)
    load_time_value = format_timing('load time', unpacked_timings['load_time'])
    sample_time_value = format_timing('sample time', unpacked_timings['sample_time'], num_runs=unpacked_timings['sample_runs'])
    prompt_eval_time_value = format_timing('prompt eval time', unpacked_timings['prompt_eval_time'], num_tokens=unpacked_timings['prompt_eval_tokens'])
    eval_time_value = format_timing('eval time', unpacked_timings['eval_time'], num_runs=unpacked_timings['eval_runs'])
    total_time_value = format_timing('total time', unpacked_timings['total_time'], num_tokens=unpacked_timings['total_tokens'])
    return {
        "load_time": load_time_value,
        "sample_time": sample_time_value,
        "prompt_eval_time": prompt_eval_time_value,
        "eval_time": eval_time_value,
        "total_time": total_time_value
    }

def print_llama_timings(timings):
    """
    Print llama timings in the desired format.

    Args:
        timings (llama_timings): Llama timings object.
    """
    prefix = "llama_print_timings: "
    timing_dict = get_timing_dict(timings)
    print(f"{prefix}        {timing_dict['load_time']}")
    print(f"{prefix}      {timing_dict['sample_time']}")
    print(f"{prefix}{timing_dict['prompt_eval_time']}")
    print(f"\033[91m{prefix}        {timing_dict['eval_time']}\033[0m")
    print(f"{prefix}       {timing_dict['total_time']}")
    return timing_dict

    # unpacked_timings = unpack_llama_timings(timings)
    # print(f"{prefix}{format_timing('        load time', unpacked_timings['load_time'])}")
    # print(f"{prefix}{format_timing('      sample time', unpacked_timings['sample_time'], num_runs=unpacked_timings['sample_runs'])}")
    # print(f"{prefix}{format_timing('prompt eval time', unpacked_timings['prompt_eval_time'], num_tokens=unpacked_timings['prompt_eval_tokens'])}")
    # print(f"{prefix}{format_timing('        eval time', unpacked_timings['eval_time'], num_runs=unpacked_timings['eval_runs'])}")
    # print(f"{prefix}{format_timing('       total time', unpacked_timings['total_time'], num_tokens=unpacked_timings['total_tokens'])}")

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "llama_cpp_pipeline"

        self.name = "Llama C++ Pipeline"
        self.llm = None
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        from llama_cpp import Llama, llama_get_timings
        # from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
        # from llama_cpp.llama_grammar import LlamaGrammar, ARITHMETIC_GBNF
        # import json
        # self.grammar = LlamaGrammar.from_string(WEB_SCHEMA, verbose=True)
        # self.llm: Llama | Callable = lambda _ : Llama(
        self.llm = Llama(
            model_path = MODEL_PATH,
            n_gpu_layers=-1, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            n_ctx=8000, # Uncomment to increase the context window
            port=1234,
            # draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10) 
            # # num_pred_tokens is the number of tokens to predict. 10 is the default.
        )
        self.timing_fn = llama_get_timings
        self.valves = self.Valves(
            **{
                "pipelines": [],
            }
        )
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass
    
    # async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
    #     print(f"pipe:{__name__}")
    
    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        try:
            assert self.llm is not None
            print("Getting times from llama.cpp!")
            timings = self.timing_fn(self.llm.ctx)
            timings_dict = print_llama_timings(timings)
            body["timings_dict"] = timings_dict
        except Exception as e:
            print("Darn! Error in getting timings")
            raise e
        
        # response_string = body["response"]
        # response_json = json.loads(response_string)
        # print(json.dumps(response_json, indent=2))
        print("Outlet flushed!")
        return body
    
    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)
        print(body)
        GRAMMAR_ENABLED = True
        if GRAMMAR_ENABLED:
            response_format = None
        else:
            self.grammar = None
            response_format = {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"team_name": {"type": "string"}},
                    "required": ["team_name"],
                },
                # "schema": Car.model_json_schema(),
            }
            # response_format = AudioFeature.model_json_schema()
            print(response_format)
        # og_response = self.llm.create_chat_completion(
        #     logit_bias='12212-inf',
        #     stop=None,
        # )
        response = self.llm.create_chat_completion_openai_v1(
            messages=messages,
            stream=body["stream"],
            # stop=["}"]
            # logit_bias='12212-inf',
            # response_format=response_format
            # grammar=self.grammar,
        )
        return response

async def main():
    """
    Example usage of the pipeline.

    This function uses a CarAndOwner schema to generate a response.
    """
    pipeline = Pipeline()
    await pipeline.on_startup()
    from constants import TEST_MESSAGES
    if True:
        messages = TEST_MESSAGES
    else:
        messages = [
            {
                "role": "user",
                "content": "Create a 3 step plan for making an oreo milkshake.",
            },
        ]
    body = {
        "stream": True,
    }
    #TODO: Get usage
    # ChatCompletionChunk(id='chatcmpl-5b36d59b-57a9-4544-b28b-b6e1b101a647', choices=[Choice(delta=ChoiceDelta(content=' ', function_call=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1722746793, model='/Users/tan/.models/lm-studio/lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q8_0.gguf', object='chat.completion.chunk', service_tier=None, system_fingerprint=None, usage=None)
    if not body["stream"]:
        print("Not streaming")
        chat_response: ChatCompletion = pipeline.pipe("Hello", "llama_cpp_pipeline", messages, body)
        response_string = chat_response.choices[0].message.content
        print(response_string)
    else:
        response_string = ""
        for chunk in pipeline.pipe("Hello", "llama_cpp_pipeline", messages, body):
            current_chunk: ChatCompletionChunk = chunk
            value = current_chunk.choices[0].delta.content
            if value is not None:
                response_string += value
                # Print the response
                print(value, end="", flush=True)
        print()
    # print(response_string)
    # Shutdown in 10 seconds
    body["response"] = response_string
    await pipeline.outlet(body)
    await pipeline.on_shutdown()

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())