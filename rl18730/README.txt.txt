Please put rl18730.py and the /rl18730/ subfolder into your /bots/ subfolder in your copy of the 'The Resistance' framework.

This is because it uses relative paths from the current working directory to load the neural network/pickled win chances/pickled sabotage chances info/knows where to log data to.

It will log important data to /bots/rl18730/GameRecord.log as well as the main 'rl18730.log' file which it would create by default, because this bot also occasionally sends messages in chat, and I did not want those to completely pollute the log file with the data used to train the neural network model.

On a related note, if you wish to train the neural network, use the jupyter notebook in the /rl18730/ subfolder, and rename GameRecord.log to 'heuristic bayes training data log.log'.