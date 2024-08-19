# modified from https://github.com/jennyzzt/LLM_debate_on_ARC
# prompt also inspired by https://github.com/rgreenblatt/arc_draw_more_samples_pub/blob/master/arc_solve/prompting.py   

import concurrent.futures
import random
import string

import numpy as np

TASK_OVERVIEW = """You will be given some number of paired example inputs and outputs grids. The outputs were produced by applying a transformation rule to the input grids. In addition to the paired example inputs and outputs, there is also one test input without a known output.
The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). Each number corresponds to a color. 0 is black.
Your task is to determine the transformation rule from examples and find out the answer, involving determining the size of the output grid for the test and correctly filling each cell of the grid with the appropriate color or number.

The transformation only needs to be unambiguous and applicable to the example inputs and the test input. It doesn't need to work for all possible inputs. Observe the examples carefully, imagine the grid visually, and try to find the pattern.
"""


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def file_to_string(filepath):
    with open(filepath, 'r') as f:
        data = f.read().strip()
    return data


def list_to_string(list_2d):
    sublists_as_strings = [f"[{','.join(map(str, sublist))}]" for sublist in list_2d]
    return f"[{','.join(sublists_as_strings)}]"


def format_arc_data(arc_data, direct=False):
    task_str = TASK_OVERVIEW

    task_demo_str = ''
    # Get task demo string
    task_demo_str += '## Examples:\n\n'
    for i, demo in enumerate(arc_data['train']):
        task_demo_str += f'### Example {i}:\n'
        task_demo_str += f'input = {list_to_string(demo["input"])}\n'
        task_demo_str += f'output = {list_to_string(demo["output"])}\n\n'

    # Get task test string
    task_test_str = ''
    for testcase in arc_data['test']:
        task_test_str += '## Test Problem:\n'
        task_test_str += f'Given input:\n {list_to_string(testcase["input"])}\n\n'
        task_test_str += f'Analyze the transformation rules based on the provided Examples and determine what the output should be for the Test Problem.'

    task_str += task_demo_str + task_test_str

    return task_str, arc_data['train'], arc_data['test'][0]['input']


def get_percentage_match(arr1, arr2):
    # arr1 is solution
    if not arr2:
        return 0
    score = 0
    for i, xs in enumerate(arr1):
        try:
            for j, x in enumerate(xs):
                try:
                    if len(arr2) > i and len(arr2[i]) > j and arr2[i][j] == x:
                        score += 1
                except:
                    pass
        except:
            pass
    score = score / (len(arr1) * len(arr1[0]))
    return score


def eval_algo(solve_fn, arc_data, soft_eval=False):
    # Calculate percentage of test cases done correctly
    testcases = arc_data['test']
    scores = []
    for testcase in testcases:
        input = testcase['input']
        output = testcase['output']
        gen_output = None
        # Run solve_fn with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(solve_fn, input)
                try:
                    gen_output = future.result(timeout=30)
                except concurrent.futures.TimeoutError:
                    future.cancel()
            except:  # if the function does not work
                continue
        # Check if correct output
        if soft_eval:
            score = get_percentage_match(output, gen_output)
        else:
            score = 1 if output == gen_output else 0
        scores.append(score)
    return np.mean(scores)


def eval_solution(output, arc_data, soft_eval=False):
    if not output:
        return 0

    solution = arc_data['test'][0]['output']
    if soft_eval:
        score = get_percentage_match(solution, output)
    else:
        score = 1 if output == solution else 0
    return score


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.
    
    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    # Return the formatted string with confidence interval and median
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"
