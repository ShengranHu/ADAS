<h1 align="center">
  <img src="./../misc/art_fig.png" width="200" /></a><br>
  <b>Automated Design of Agentic Systems</b><br>
</h1>

### This example uses Gemini 1.5 Flash free tier.

```Puzzles now display in terminal during evaluation```
<h1 align="left">
  <img src="gemini_arc/misc/terminal_logs2.png" width="600" /></a><br>
</h1>



## Gemini Model API 

### Get your free API Key:   
<a href="https://ai.google.dev/gemini-api">https://ai.google.dev/gemini-api</a>


<p align="left">
<img src="gemini_arc/misc/gemini_key.png"  width="600"/></a><br>
</p>

### API Free Limits
```
Free Tier RATE LIMITS  

Gemini 1.5 Flash
15 RPM (requests per minute)
1 million TPM (tokens per minute)
1,500 RPD (requests per day)
- free context caching, up to 1 million tokens of storage per hour
```
Full details:
https://aistudio.google.com/app/plan_information


## Setup
```bash
# Install libraries
# From the ADAS/examples directory run
pip install -r requirements.txt

# Add your Gemini API key to the example.env file
GEMINI_API_KEY=""

# Rename the example.env file to .env'
example.env -> .env
```

## Running Instructions

### Running Meta Agent Search with Gemini

Navigate to ADAS/examples folder. Launch experiment using the `search.py` script.

```bash
# From the ADAS/examples directory run
python gemini_arc/search.py
```

### Customizing Meta Agent Search for New Domains

You can easily adapt the code to search for new domains. To do so, follow these steps:

1. Modify the `evaluate_forward_fn()` function and adjust any necessary formatting prompts in the `search.py` file. 

2. Consider adding additional basic functions for the meta agent to utilize during the design process.

3. Update the domain-specific information within the prompts to match the requirements of your new domain.

4. Run the search and evaluation on your new domain.

### Safety Consideration
> [!WARNING]  
> The code in this repository involves executing untrusted model-generated code. We strongly advise users to be aware of this safety concern. While it is highly unlikely that model-generated code will perform overtly malicious actions in our current settings and with the models we use, such code may still act destructively due to limitations in model capability or alignment. By using this repository, you acknowledge and accept these risks.


## Citing
If you find this project useful, please consider citing:
```
@article{hu2024ADAS,
title={Automated Design of Agentic Systems},
author={Hu, Shengran and Lu, Cong and Clune, Jeff},
journal={arXiv preprint arXiv:2408.08435},
year={2024}
}
```
