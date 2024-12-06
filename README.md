# CS 5060 Course Project
## Group Members
- Brandon Herrin (A02336477)
- Josh Weeks (A02304519)

## Cloning the Code Repository
SSH:
```bash
git clone git@github.com:CodyRooBoy/cs5060-project.git cs5060-project
```
HTTPS: 
```bash
git clone https://github.com/CodyRooBoy/cs5060-project.git cs5060-project
```

## Running the Code
### Create a Virtual Environment
```bash
python -m venv env
```

### Activate the Virtual Environment
Windows:
```bash
. env/Scripts/activate
```

Linux:
```bash
source env/bin/activate`
```

### Install Python Dependencies
`pip install -r requirements.txt`

## Run The Code

### If you would like to see how we did our hyperparameter search
-  Run `python hyperparameter_search.py`

This will do a grid search varying the learning rate between 0.001, 0.0005, and 0.01, varying the gamma between 0.99, 0.98 and 0.9, and varying the batch size between 32, 64. The logs of all the runs will be saved to /dqn_tensorboard_logs/{model_name}/, and the graphs of the results of the model playing against the "random agent" will be saved to "result_graphs". 

#### Launching Tensorboard
If you want to explore the runs along with interactive graphs of each model that is being evaluated, you can run:
- `tensorboard --logdir=dqn_tensorboard_logs/`


### If you would like to see an example run of a model playing against a random agent
-  Run `python run-game.py`