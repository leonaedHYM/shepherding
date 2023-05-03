
## Usage
Operate the model:

* python run.py -t -r -p
    * -p enter the evaluation mode
    * -t trains the existing model
    * -t -r trains the model from scratch

## Tips:
1. ./Envs folder includes two multi-shepherd environments totally.  The multi_agent_Env2.py is built up based on the Strombom Model. The multi_agent_environment.py is built up based on the Reynold Dynamics. Each environment can be used by the model. 

2. Model structure details are defined in the model.py. 

3. Some pretrained model were stored in the ./model folder. And the other model you trained will also be stored in this folder. Remember to change the checkpoint name in the parameters.py!





