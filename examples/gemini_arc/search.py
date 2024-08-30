import random
import argparse
import copy
import json
import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np

from tqdm import tqdm
from colorama import Fore, Style, init

import time
import sys
import logging
from datetime import datetime
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from arc_prompt import get_init_archive, get_prompt, get_reflexion_prompt
from utils import visualize_image_using_emoji, random_id, format_arc_data, eval_solution, list_to_string, bootstrap_confidence_interval
from utils import setup_logging, api_call_with_rate_limit, check_max_api_calls, visualization_logger

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""# Output Format:\nReply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a WELL-FORMED JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}.\n\n"
SYSTEM_MSG = ""
CODE_INST = "You will write code to solve this task by creating a function named `transform`. This function should take a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation that works for both example and test inputs. Make sure that the transform function is capable of handling both example and test inputs effectively, reflecting the learned transformation rules from the Examples inputs and outputs."

SEARCHING_MODE = True

# Global API tracker
api_tracker = {
    'api_call_count': 0,
    'total_api_call_count': 0,
    'last_reset_time': time.time()
}

log_file = f"gemini_arc/logs/arc_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
visualization_log_file = f"gemini_arc/logs/arc_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

@backoff.on_exception(backoff.expo, Exception)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
):
    global api_tracker
    api_tracker = api_call_with_rate_limit(api_tracker, args.max_api_calls)
    
    logging.info(f"API call #{api_tracker['api_call_count']} to Gemini")
    
    model = genai.GenerativeModel(
        model_name=model,   
        generation_config={
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json"
        }
    ) 
    
    #logging.info(f"system_message: {system_message}")
    #logging.info(f"msg: {msg}")
    
    prompt = system_message + msg    
    response = model.generate_content(prompt)
    content = response.text
    logging.info(f"Response from Gemini: {content}")
    try:
        json_dict = json.loads(content)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON: {content}")
        json_dict = {}  # Return an empty dictionary on error
    assert not json_dict is None
    return json_dict

@backoff.on_exception(backoff.expo, Exception)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8
):
    global api_tracker
    api_tracker = api_call_with_rate_limit(api_tracker, args.max_api_calls)
    
    logging.info(f"API call #{api_tracker['api_call_count']} to Gemini (reflect)")
    
    model = genai.GenerativeModel(
        model_name=model,   
        generation_config={
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json"
        }
    ) 

    prompt = ""

    for msg in msg_list:
        prompt += f"{msg['role']}: {msg['content']}\n"
    #logging.info(f"Prompt_reflect: {prompt}")
       
    response0 = model.generate_content(prompt)
    content0 = response0.text
    logging.info(f"Response from Gemini (reflect): {content0}")

    try:
        json_dict = json.loads(content0)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON: {content0}")
        
        api_tracker = api_call_with_rate_limit(api_tracker, args.max_api_calls)
        logging.info(f"API call #{api_tracker['api_call_count']} to Gemini (reflect) fix JSON")
        try:
            prompt2 = f"""You are a JSON formatting expert. Your task is to fix and properly format the following JSON object. Please follow these steps:

                1. Identify and correct any syntax errors in the JSON structure.
                2. Ensure all keys and string values are properly enclosed in double quotes.
                3. Escape any special characters within string values, such as newlines or quotes.
                4. Format the JSON with proper indentation for readability.
                5. If there are any unintended line breaks within string values, replace them with '\n'.
                6. Make sure all brackets and braces are properly closed and balanced.

                Here's the JSON object to fix:

                json
                {content0}
                
                Please provide the corrected and properly formatted JSON.
                """
                    
            response = model.generate_content(prompt2)
            content = response.text
            json_dict = json.loads(content)
        except json.JSONDecodeError:
            logging.error(f"Error decoding fixed JSON: {content}")
            json_dict = {}

    assert json_dict is not None
    return json_dict

class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gemini-1.5-flash', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        code_output = False

        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." for key in self.output_fields}
        for key in output_fields_and_description:
            if 'answer' in key:
                output_fields_and_description[key] = f"Your {key}. ONLY return a string of list[list[int]]. DO NOT return anything else."
            elif 'code' in key:
                output_fields_and_description[key] = f"Your {key}. Don't write tests in your Python code, ONLY return the `transform` function. DO NOT return anything else. (It will be tested later.)"
                code_output = True
        system_prompt = ROLE_DESC(self.role) + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue

            if isinstance(content, list):
                try:
                    content = list_to_string(content)
                except:
                    pass

            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + "# Instruction: \n" + instruction + "\n\n" + (CODE_INST if code_output else '')
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            reponse_json = {}
            reponse_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(reponse_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            logging.info(f"Query Error: {e}")
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in reponse_json and len(reponse_json) < len(self.output_fields):
                    reponse_json[key] = ''
            for key in copy.deepcopy(list(reponse_json.keys())):
                if len(reponse_json) > len(self.output_fields) and not key in self.output_fields:
                    del reponse_json[key]
        output_infos = []
        for key, value in reponse_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self, examples, test_iuput) -> None:
        self.examples = examples
        self.test_iuput = test_iuput

    def run_examples_and_get_feedback(self, code):
        examples = self.examples

        correct_examples = []
        wrong_examples = []

        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        gen_output = lambda msg: Info('feedback', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}"), correct_examples, wrong_examples
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code."), correct_examples, wrong_examples

        transform = local_vars['transform']

        feedback = ""

        for idx, example in enumerate(examples):
            input_grid = example['input']
            output_grid = example['output']
            try:
                transformed_grid = transform(input_grid)
            except Exception as e:
                return gen_output("Error during function execution: {e}"), correct_examples, wrong_examples

            if transformed_grid == output_grid:
                feedback += f"Your transform function generates a CORRECT answer in Example {idx}!\n\n"
                correct_examples.append(example)
            else:
                try:
                    transformed_grid = list_to_string(transformed_grid)
                except:
                    pass
                feedback += f"Your transform function generates a WRONG answer in Example {idx}!\nExpect: See above Example {idx} output.\nYou got: {transformed_grid}\nObserve the Example {idx} carefully!\n\n"
                wrong_examples.append(example)

        return gen_output(feedback), correct_examples, wrong_examples

    def get_test_output_from_code(self, code):
        test_input = self.test_iuput

        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        gen_output = lambda msg: Info('answer', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}")
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code.")

        transform = local_vars['transform']
        try:
            transform_output = transform(test_input)
            transform_output = list_to_string(transform_output)
            #logging.info(f"transform_output: {transform_output}")
        except Exception as e:
            logging.info(f"Error during function execution: {e}")
            return gen_output("Error during function execution: {e}")

        return gen_output(transform_output)

def search(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    for solution in archive:
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        logging.info(f"============Initial Archive: {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"])
        except Exception as e:
            logging.info("During evaluating initial archive:")
            logging.info(f"Error: {e}")
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['fitness'] = fitness_str

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        logging.info(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            logging.info(f"next_solution0: {next_solution}")

            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            # Reflexion 1
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            logging.info(f"next_solution1: {next_solution}")
            # Reflexion 2
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            logging.info(f"next_solution3: {next_solution}")
        except Exception as e:
            logging.info("During LLM generate new solution:")
            logging.info(f"Error: {e}")
            continue

        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                logging.info("During debug evaluation:")
                logging.info(f"Error: {e}")
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                    logging.info(f"next_solution_debug: {next_solution}")
                except Exception as e:
                    logging.info("During LLM generate new solution:")
                    logging.info(f"Error: {e}")
                    continue
                continue
        if not acc_list:
            continue
        
        print(f"Created New Solution: {next_solution['name']} \n")
        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1
        


        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)


def evaluate(args):
    logging.info("Starting evaluation process")
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    logging.info(f"eval_file: {eval_file_path}")
    
    # Check if run archive exists
    if not os.path.exists(file_path):
        logging.error(f"Run archive file not found: {file_path}")
        return

    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)

    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)
    
    current_idx = len(eval_archive)

    # We only evaluate the solutions that have not been evaluated yet
    while current_idx < len(archive):
        sol = archive[current_idx]
        logging.info(f"Evaluating solution {current_idx + 1}/{len(archive)} (generation: {sol['generation']})")
        print(f"Evaluating solution {sol['name']}")
        
        try:
            
            acc_list = evaluate_forward_fn(args, sol["code"])
            fitness_str = bootstrap_confidence_interval(acc_list)
            logging.info(f"Evaluation successful. Fitness: {fitness_str}")
            
            sol['test_fitness'] = fitness_str
            eval_archive.append(sol)

            # Save results after each successful evaluation
            with open(eval_file_path, 'w') as json_file:
                json.dump(eval_archive, json_file, indent=4)
            logging.info(f"Updated eval archive saved. Total evaluated: {len(eval_archive)}")
        
        except Exception as e:
            logging.error(f"Error evaluating solution {current_idx}: {e}")
        
        current_idx += 1

def evaluate_forward_fn(args, forward_str):
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)
            
    if SEARCHING_MODE:
        arc_dir = os.path.join(os.path.expanduser("~"), "ADAS", "dataset", "ARC-800-tasks", "training")
    else:
        arc_dir = os.path.join(os.path.expanduser("~"), "ADAS", "dataset", "ARC-800-tasks", "evaluation")

    arc_data_queue = []
    file_names = []

    # Get all JSON files in the directory
    all_files = [f for f in os.listdir(arc_dir) if f.endswith('.json')]

    # Randomly sample the specified number of files
    selected_files = random.sample(all_files, min(args.num_puzzles, len(all_files)))

    for file_name in selected_files:
        file_path = os.path.join(arc_dir, file_name)
        with open(file_path, 'r') as json_file:
            arc_data = json.load(json_file)
            arc_data_queue.append((arc_data, file_name))  # Store file_name with arc_data
            file_names.append(file_name)

    logging.info(f"# of Puzzles: {len(arc_data_queue) * args.n_repreat}")
    max_workers = min(len(arc_data_queue) * args.n_repreat, args.max_workers) if args.multiprocessing else 1
    
    agent_task_queue = []
    for arc_data, file_name in arc_data_queue:
        task_str, examples, test_input = format_arc_data(arc_data)
        taskInfo = Info('task', 'User', task_str, -1)
        #logging.info(f"taskInfo for {file_name}: {taskInfo}")
        agent_task_queue.extend([(AgentSystem(examples, test_input), taskInfo, arc_data, file_name)] * args.n_repreat)

    def call_forward(agent_task):
        agent, taskInfo, arc_data, file_name = agent_task
        res = agent.forward(taskInfo)
        origin_res = res
        try:
            if isinstance(res, Info):
                res = res.content

            if isinstance(res, str):
                res = eval(res)
                logging.info(f"Our Answer: {res}")
        
            correct_answer = arc_data["test"][0]["output"]
            logging.info(f"Correct Answer: {correct_answer}")
        
            json_file_name = file_name # "test_name" 
        
            input_grid = arc_data["test"][0]["input"]
            visualization = visualize_image_using_emoji(input_grid, res, correct_answer, 
                                                        titles=["Input", "Output", "Correct"])
                    
            if args.print_visuals:
                # Print only the puzzle name and visualization to the terminal
                print(f"\nPuzzle: {json_file_name}")
                print(visualization + "\n")
        
            if args.log_visuals:
                # Log the visualization with the JSON file name
                visualization_logger.info(f"\nPuzzle: {json_file_name}\nVisualization:\n{visualization}")
                #visualization_logger.info(f"Visualization:\n{visualization}")
        
            logging.info("Evaluating the solution")
            hard_score = eval_solution(res, arc_data, soft_eval=False)
            logging.info(f"hard_score: {hard_score}")
            return hard_score
        except Exception as e:
            logging.error(f"Error during evaluation for {file_name}: {e}")
            return 0

    # Initialize colorama for Windows compatibility
    #init()

    # Create a colorful tqdm progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        acc_list = list(tqdm(
            executor.map(call_forward, agent_task_queue),
            total=len(agent_task_queue),
            desc=f"{Fore.CYAN}Evaluating puzzles{Style.RESET_ALL}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            colour='green'
        ))

    # Print a summary of the results
    correct = sum(acc_list)
    total = len(acc_list)
    accuracy = correct / total * 100

    print(f"\n{Fore.YELLOW}Summary:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Correct: {correct}{Style.RESET_ALL}")
    print(f"{Fore.RED}Incorrect: {total - correct}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Accuracy: {accuracy:.2f}%{Style.RESET_ALL}")

    logging.info(f"accuracy:  {bootstrap_confidence_interval(acc_list)}")
    return acc_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=1)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='gemini_arc/results/')
    parser.add_argument('--expr_name', type=str, default='arc_gemini_results')
    parser.add_argument('--n_generation', type=int, default=1)
    parser.add_argument('--reflect_max', type=int, default=1)
    parser.add_argument('--debug_max', type=int, default=1)
    parser.add_argument('--model', type=str, default='gemini-1.5-flash', choices=[])
    parser.add_argument('--log_info', action='store_true', default=True, help='Enable info logging')
    parser.add_argument('--log_visuals', action='store_true', default=True, help='Enable logging of visualizations')
    parser.add_argument('--print_visuals', action='store_true', default=True, help='Print visualizations to terminal')
    parser.add_argument('--num_puzzles', type=int, default=10, help='Number of puzzles to process')
    parser.add_argument('--max_api_calls', type=int, default=100, help='Maximum number of API calls')
    
    args = parser.parse_args()
    
    visualization_logger = setup_logging(args, log_file, visualization_log_file)
    if args.log_info:
        logging.info(f"Starting script with args: {args}")
    
    # search
    SEARCHING_MODE = True
    print("\nStarting search for new ARC solutions\n")
    if args.log_info:
        print("Details being logged\n")
    search(args)
    print("Search completed")
    if args.log_info:
        logging.info(f"Search completed. Total API calls: {api_tracker['total_api_call_count']}")
   
    # evaluate
    SEARCHING_MODE = False
    print("Starting evaluation \n")
    evaluate(args)
    print(f"Evaluation completed. Total API calls: {api_tracker['total_api_call_count']}")
    if args.log_info:
        logging.info(f"Evaluation completed. Total API calls: {api_tracker['total_api_call_count']}")
        logging.info("Script execution completed")
   
   