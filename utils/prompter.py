"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)

        return res
    def generate_prompt_tag(
        self,
        tag: str,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.

        if (tag == '0'):
            if input:
                res = self.template["prompt_input"].format(
                    instruction=instruction, input=input
                )
            else:
                res = self.template["prompt_no_input"].format(
                    instruction=instruction
                )
            if label:
                res = f"{res}{label}"
            if self._verbose:
                print(res)
        elif (tag == '1'):
            if input:
                res = self.template["prompt_cbnu_input"].format(
                    instruction=instruction, input=input
                )
            else:
                res = self.template["prompt_cbnu_no_input"].format(
                    instruction=instruction
                )
            if label:
                res = f"{res}{label}"
            if self._verbose:
                print(res)
            
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

    def generate_multiturn_prompt(self, sources: []):

        # human,gpt
        ret = ''
        for i, sentence in enumerate(sources):
            if(sentence["from"] == "human"):
                s = "사용자: " + sentence["value"] + ' '
                ret += s
            if(sentence["from"] =="gpt"):
                s = "ASSITANT: " + sentence["value"] + '</s> '
                ret += s
        return ret

        