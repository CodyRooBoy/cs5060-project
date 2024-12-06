# CS 5060 Course Project
## Group Members
- Brandon Herrin (A02336477)
- Josh Weeks (A02304519)

# Running the Code
### Create a virtual environment
`python -m venv env`

### Activate virtual environment
Windows:
- `. env/Scripts/activate`

Linux:
- `source env/bin/activate`

### Install dependencies
`pip install -r requirements.txt`

## Run the code

### If you would like to see how we did our hyperparameter search
-  Run `python hyperparameter_search.py`

This will do a grid search varying the learning rate between 0.001, 0.0005, and 0.01, varying the gamma between 0.99, 0.98 and 0.9, and varying the batch size between 32, 64. The logs of all the runs will be saved to /dqn_tensorboard_logs/{model_name}/, and the graphs of the results of the model playing against the "random agent" will be saved to "result_graphs". 

If you want to explore the runs along with interactive graphs of each model that is being evaluated, you can run:
- `tensorboard --logdir=dqn_tensorboard_logs/`


### If you would like to see an example run of a model playing against a random agent
-  Run `python run-game.py`