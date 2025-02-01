# OVERTHINK_public

This is an official repository of the paper *Overthinking: Slowdown Attacks on Reasoning LLMs*. 

### Context Agnostic
* These are jupyter notebook that allows you to reproduce are results on o1. For every test, add your openAI key in the notebook

1. `context-agnostic-o1.ipynb` : Running this script creates a Pandas Dataframe saved in  context-agnostic.pkl. Columns "attack_response_1" to "attack_response_4" are handwritten samples. Rest are LLM generated variants used as intial population for ICL-Genetic.

2. `context-agnostic-icl-o1.ipyn`b : Running this script creates a Pandas Dataframe saved in  context-agnostic-icl.pkl, saving the best performing context and its responses.

### Context Aware

1. `create_context_json.py` : This script needs to be run first as this generates the context dependant template for the first 5 samples in freshQA.

2. `context-aware-o1.ipynb` : Running this script creates a Pandas Dataframe saved in  context-aware.pkl. Columns "attack_response_1" is handwritten sample. Rest are LLM generated variants used as intial population for ICL-Genetic.

3. `context-aware-icl-o1.ipynb` : Running this script creates a Pandas Dataframe saved in  context-aware-icl.pkl, saving the best performing context and its responses.

### Checking Results

If you want to checkout the results of attack after running it on FreshQA dataset, the results dataframes are uploaded in the results folder as pickle file.
