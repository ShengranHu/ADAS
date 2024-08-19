import random
import re
import string
import xml.etree.ElementTree as ET

import numpy as np


def score_fn(target: str, prediction: str) -> bool:
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")

    target = target.replace(",", "")
    prediction = prediction.replace(",", "")

    return target == prediction


def extract_number(text):
    match = re.search(r'\d+', text)
    number = match.group() if match else None
    return number


def get_all_examples(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    examples = []
    for problem in root.find('ProblemSet').findall('Problem'):
        problem_data = {
            'ID': problem.get('ID'),
            'Grade': problem.get('Grade'),
            'Source': problem.get('Source'),
            'Body': problem.find('Body').text,
            'Question': problem.find('Question').text,
            'Solution-Type': problem.find('Solution-Type').text,
            'Answer': problem.find('Answer').text,
            'Formula': problem.find('Formula').text
        }
        if problem_data['Grade'] not in ['3', '4']:
            continue
        problem_data['inputs'] = "Solve this math problem:\n" + problem_data['Body'] + '\n' + problem_data['Question']
        number = extract_number(problem_data['Answer'])
        if not number:
            continue
        problem_data['targets'] = number
        examples.append(problem_data)

    return examples


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


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
