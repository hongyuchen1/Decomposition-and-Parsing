import json
import re
from operator import itemgetter
from typing import List, Tuple, Dict
from openai.lib._parsing._completions import type_to_response_format_param

from langchain.output_parsers import OutputFixingParser, RetryWithErrorOutputParser
from langchain_core.prompt_values import StringPromptValue
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from schema import CriterionWDecomposedTrait, CriterionWTraits, TraitDecomposed, BaseCriterion
from prompts import system_prompt, criterion_prompts_w_system, criterion_prompts, trait_prompts, \
    trait_prompts_w_system

class DPProcessor:
    def __init__(self, criterion: BaseCriterion, model: str, max_tokens: int, api_key: str = "1"):
        in_or_ex = "Exclusion" if criterion.in_or_ex == "EX " else "Inclusion"
        self.criterion = criterion
        self.og_criteria = f"{in_or_ex} Criteria:\n\n* {criterion.original_criterion}"
        self.model = model
        self.api_key = api_key
        if "gpt" in model:
            self.llm = ChatOpenAI(
                            model=model,
                            openai_api_key=api_key,
                            max_tokens=max_tokens,
                            temperature=0
                            )
        else:
            self.llm = ChatOpenAI(
                            model=model,
                            openai_api_key=api_key,
                            max_tokens=max_tokens,
                            temperature=0,
                            openai_api_base="http://localhost:8000/v1"
                            )
        final_parser = PydanticOutputParser(pydantic_object=CriterionWDecomposedTrait)
        self.final_output_format = final_parser.get_format_instructions()[3:]
        self.system_msg = ("system", system_prompt)
        self.input_w_history = {'history': itemgetter('history'), 'response': itemgetter('response'),
                          'og_criteria': lambda x: self.og_criteria,
                          'output_format': lambda x: self.final_output_format}
        self.split_criterion_chain = self.init_sub_chain(CriterionWTraits, criterion_prompts, criterion_prompts_w_system, max_tokens)
        self.decompose_trait_chain = self.init_sub_chain(TraitDecomposed, trait_prompts, trait_prompts_w_system, max_tokens)


    @staticmethod
    def fix_json_response(response) -> str:
        response = response.content
        try:
            json.loads(response)
        except json.JSONDecodeError:
            pass
        
        response = response.replace('True', 'true').replace('False', 'false')
        response = re.sub(r'```json\n|```|json', '', response)
        response = response.replace('“', '"').replace('”', '"')
        response = re.sub(r'(\d+)/(\d+)', lambda m: str(float(m.group(1)) / float(m.group(2))), response)
        response = response.strip()
        
        match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', response)
        cleaned_string = match.group(0) if match else response
        
        if (cleaned_string.startswith('[') and not cleaned_string.endswith(']')) or (cleaned_string.startswith(
                '{') and not cleaned_string.endswith('}')):
            if cleaned_string.count('{') == cleaned_string.count('}') and cleaned_string.count(
                    '[') == cleaned_string.count(']'):
                cleaned_string = cleaned_string[:-2] + cleaned_string[-1] + cleaned_string[-2]
            else:
                cleaned_string = cleaned_string[:-1]
        
        open_curly, close_curly = cleaned_string.count('{'), cleaned_string.count('}')
        open_square, close_square = cleaned_string.count('['), cleaned_string.count(']')
        
        if open_curly == 1 and close_curly == 0:
            cleaned_string += '}'
        elif close_curly == 1 and open_curly == 0:
            cleaned_string = '{' + cleaned_string
        elif open_square == 1 and close_square == 0:
            start = 0
            while cleaned_string.find('}', start) != -1:
                try:
                    loc = cleaned_string.find('}', start)
                    json.loads(cleaned_string[:loc + 1] + ']' + cleaned_string[loc + 1:])
                    return cleaned_string[:loc + 1] + ']' + cleaned_string[loc + 1:]
                except json.JSONDecodeError:
                    start = loc + 1
        elif close_square == 1 and open_square == 0:
            cleaned_string = '[' + cleaned_string
        
        if open_curly == 0 and close_curly == 0 and open_square == 0 and close_square == 0:
            cleaned_string = '{' + cleaned_string + '}'
        
        try:
            json.loads(cleaned_string)
            return cleaned_string
        except json.JSONDecodeError:
            if open_curly > close_curly or open_square > close_square:
                cleaned_string = cleaned_string[1:]
            elif open_curly < close_curly or open_square < close_square:
                cleaned_string = cleaned_string[:-1]
                
            try:
                json.loads(cleaned_string)
                return cleaned_string
            except json.JSONDecodeError:
                cleaned_string = cleaned_string.replace("'s", "<s_holder>").replace("'re", "<are_holder>")
                cleaned_string = cleaned_string.replace("'", '"').replace("\n", " ").replace("\t", " ")
                cleaned_string = cleaned_string.replace("<s_holder>", "'s").replace("<are_holder>", "'re")
            
                try:
                    json.loads(cleaned_string)
                    return cleaned_string
                except json.JSONDecodeError:
                    try:
                        wrapped_string = f"[{cleaned_string}]"
                        json.loads(wrapped_string)
                        return wrapped_string
                    except json.JSONDecodeError:
                        print(f"Unable to fix JSON response: {cleaned_string}")
                        return cleaned_string

    @staticmethod
    def get_historic_prompt(result) -> str:
        history = result.messages[-1].content
        if "**Step " in history:
            histories = history.split("**Step ")
            steps = histories[-1].split("\n")
            return f"{'**Step '.join(histories[:-1])}**Step {steps[0]}"
        else:
            return history

    def build_chains(self, prompts:List[str], prompts_w_system_prompt:List[bool], output_format:str) \
            -> List[Tuple[Dict, ChatPromptTemplate]]:
        chains = []
        for i, (prompt, w_system_prompt) in enumerate(zip(prompts, prompts_w_system_prompt)):
            if w_system_prompt:
                chat_prompt = ChatPromptTemplate.from_messages([self.system_msg, ("human", prompt)])
            else:
                chat_prompt = ChatPromptTemplate.from_messages([("human", prompt)])
            if i == 0:
                chains.append(({"og_text": itemgetter("og_text"),
                                "og_criteria": lambda x: self.og_criteria,
                                'output_format': lambda x: self.final_output_format} |
                        chat_prompt))
            elif i == len(prompts) - 1:
                chains.append(({**self.input_w_history, 'format_instructions': lambda x: output_format} |
                        chat_prompt))
            else:
                chains.append((self.input_w_history | chat_prompt))

        return chains

    def iter_chains(self, m_input, chains):
        prompt = chains[0].invoke(m_input)
        result = self.llm.invoke(prompt)
        prompt = self.get_historic_prompt(prompt)
        for chain in chains[1:]:
            prompt = chain.invoke({'history': prompt, 'response': result.content})
            result = self.llm.invoke(prompt)
            prompt = self.get_historic_prompt(prompt)
        return {"prompt_value": StringPromptValue(text=prompt), "completion": result}

    def init_sub_chain(self, schema, prompts:List[str], add_system:List[bool], max_tokens:int):
        if "gpt" in self.model:
            structured_llm = ChatOpenAI(
                    model=self.model,
                    openai_api_key=self.api_key,
                    max_tokens=max_tokens,
                    temperature=0,
                    extra_body={
                        "response_format":type_to_response_format_param(CriterionWDecomposedTrait)
                    }
                )
            
        else:
            structured_llm = ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key,
                max_tokens=max_tokens,
                temperature=0,
                openai_api_base="http://localhost:8000/v1",
                extra_body={
                    "guided_json": schema.model_json_schema(),
                }
            )
        output_parser = PydanticOutputParser(pydantic_object=schema)
        fix_parser = OutputFixingParser.from_llm(structured_llm, parser=output_parser)
        retry_parser = RetryWithErrorOutputParser.from_llm(structured_llm, parser=fix_parser, max_retries=3)
        output_format = output_parser.get_format_instructions()

        chains = self.build_chains(prompts,
                                        add_system,
                                            output_format)

        component_chain = (RunnableLambda(lambda x: self.iter_chains(x, chains))
                       | RunnableLambda(lambda x: {"prompt_value": x["prompt_value"],
                                                   "completion": self.fix_json_response(x["completion"])})
                       | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x)))
        return component_chain

    def run(self) -> CriterionWDecomposedTrait:
        criteria_w_traits = self.split_criterion_chain.invoke({'og_text': json.dumps(self.criterion.dict()), "og_criteria": self.og_criteria})
        traits = self.decompose_trait_chain.batch([{'og_text': json.dumps(trait.dict()), "og_criteria": self.og_criteria} for trait in criteria_w_traits.traits])
        criteria_w_traits.traits = traits
        return criteria_w_traits


def main(criterion: str, in_or_ex: int|bool, model:str="meta-llama/Llama-3.3-70B-Instruct", max_token:int=8192, api_key:str="1"):
    for i in range(3):
        try:
            processor = DPProcessor(BaseCriterion(index=1, in_or_ex="IN" if in_or_ex else "EX", original_criterion=criterion), model=model, max_tokens=max_token, api_key=api_key)
            final_output = processor.run()
            return final_output
        except Exception as e:
            print(e)
    print('Criterion Fail: ', criterion)


if __name__ == "__main__":
    converted_result = main("Patient has a history of diabetes.", 1)
    print(converted_result)