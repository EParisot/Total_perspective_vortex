import os
import click
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

from train import get_dataset, get_Xy

DATA_DIR = "data"
MODEL_NAME = "TPV_pipeline.joblib"

@click.command()
@click.argument('data_path', type=click.Path(exists=True), default=DATA_DIR)
@click.option('-s', '--subjects', required=False, type=int, default=1)
@click.option('-r', '--runs', required=False, type=click.Choice(['real', 'imaginary', 'both'], case_sensitive=False), default="both")
@click.option('-m', '--model', type=click.Path(exists=True), default=MODEL_NAME)
@click.option('-v', '--verbose', is_flag=True)
def main(data_path, subjects, runs, model, verbose):

	pipeline = joblib.load(model)

	if subjects <= 0 or subjects > 109:
		print("ERROR: Invalid Subject %d" % subjects)
		exit(0)
	# get data
	dataset = get_dataset(data_path, [subjects], runs, apply_filter=True, verbose=verbose)
	# parse data
	X, y = get_Xy(dataset)
	# set CV
	n_splits = 5  # how many folds to use for StratifiedKFold
	cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

	sfreq = X.info["sfreq"]
	print("sfreq= ", sfreq)
	X = X.get_data()

	print("X shape= ", X.shape, "y shape= ", y.shape)
	
	scores = []
	for n in range(X.shape[0]):

		pred = pipeline.predict(X[n:n+1, :, :])
		print("pred= ", pred, "truth= ", y[n:(n + 1)])
		scores.append(1 - np.abs(pred[0] - y[n:(n + 1)][0]))
	print("Mean acc= ", np.mean(scores))

if __name__ == "__main__":
	main()