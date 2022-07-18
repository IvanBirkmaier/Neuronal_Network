.PHONY: test
test:
	python test.py

.PHONY: data
data:
	python dataService\getFeatureVariables.py

.PHONY: runLSTM
runLSTM:
	python LSTM\LSTM_Pytorch.py
