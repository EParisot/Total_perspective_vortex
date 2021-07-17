import os
import click

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np
import mne
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, cross_val_score
import joblib

from CSP import CSP

DATA_DIR = "data"
SUBJECTS = [1] # 1 indexed

R1_RUNS = [6, 10, 14] # hands versus foots real
I1_RUNS = [5, 9, 13] # hands versus foots imaginary
R2_RUNS = [4, 8, 12] # left vs right hand real
I2_RUNS = [3, 7, 11] # left vs right hand imaginary


save_name = "TPV_pipeline.joblib"

def get_dataset(data_path, task, subjects, runs, apply_filter=True, verbose=True):
	# get subjects files names
	raw_fnames = {}
	for i, d in enumerate(os.listdir(data_path)):
		if os.path.isdir(os.path.join(data_path, d)) and i + 1 in subjects:
			raw_fnames[d] = os.listdir(os.path.join(data_path, d))
	# for subjects, read runs
	_runs = []
	if task == "handsvsfoot" or task == "both":
		if runs == "real" or runs == "both":
			_runs += R1_RUNS
		if runs == "imaginary" or runs == "both":
			_runs += I1_RUNS
	if task == "leftvsrighthand" or task == "both":
		if runs == "real" or runs == "both":
			_runs += R2_RUNS
		if runs == "imaginary" or runs == "both":
			_runs += I2_RUNS
	dataset = []
	sfreq = None
	for d in raw_fnames:
		subject = []
		b = False
		for i, f in enumerate(raw_fnames[d]):
			if f.endswith(".edf") and int(f.split('R')[1].split(".")[0]) in _runs:
				subject_data = mne.io.read_raw_edf(os.path.join(data_path, d, f), preload=True)
				if sfreq == None:
					sfreq = subject_data.info["sfreq"]
				# check we got consistant freq across runs
				if subject_data.info["sfreq"] == sfreq:
					subject.append(subject_data)
				else:
					b = True
					print("WARNING: sample freqency inconsistency detected, frames have been droped")
					break
		if b:
			continue
		dataset.append(mne.concatenate_raws(subject))
	# concat every raws
	dataset = mne.concatenate_raws(dataset)
	# parse data
	dataset.rename_channels(lambda s: s.strip("."))
	montage = mne.channels.make_standard_montage("standard_1020")
	mne.datasets.eegbci.standardize(dataset)  # set channel names
	dataset.set_montage(montage)
	# low / high cut filter
	if apply_filter:
		dataset.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')

	if verbose:
		montage.plot()
		print(dataset)
		print(dataset.info)
		print(dataset.info["ch_names"])
		print(dataset.annotations)
		mne.viz.plot_raw(dataset, scalings={"eeg": 75e-6}, block=True)

	return dataset

def get_Xy(dataset):
	event_id = dict(T1=0, T2=1)
	# avoid classification of evoked responses by using epochs that start 1s after cue onset.
	tmin, tmax = -1., 4.
	events, _ = mne.events_from_annotations(dataset, event_id=event_id)
	picks = mne.pick_types(dataset.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
	epochs = mne.Epochs(dataset, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)   
	labels = epochs.events[:, -1]
	return epochs, labels

@click.command()
@click.argument('data_path', type=click.Path(exists=True), default=DATA_DIR)
@click.option('-t', '--task', type=click.Choice(['handsvsfoot', 'leftvsrighthand'], case_sensitive=False), default="handsvsfoot")
@click.option('-s', '--subject', required=False, type=int, default=1)
@click.option('-r', '--runs', required=False, type=click.Choice(['real', 'imaginary', 'both'], case_sensitive=False), default="both")
@click.option('-v', '--verbose', is_flag=True)
def main(data_path, task, subject, runs, verbose):
	if subject <= 0 or subject > 109:
		print("ERROR: Invalid Subject %d" % subject)
		exit(0)
	# get data
	dataset = get_dataset(data_path, task, [subject], runs, apply_filter=True, verbose=verbose)
	# parse data
	X, y = get_Xy(dataset)
	# set CV
	n_splits = 5  # how many folds to use for StratifiedKFold
	cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	# classifer
	lda = LinearDiscriminantAnalysis()
	# own CSP
	csp = CSP(n_components=4)
	# scaler
	sca = mne.decoding.Scaler(X.info)
	X = X.get_data()
	# build pipeline
	pipeline = Pipeline([('SCA', sca), ('CSP', csp), ('LDA', lda)])
	# run training
	scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=1)
	# printing the results
	class_balance = np.mean(y == 0)
	class_balance = max(class_balance, 1. - class_balance)
	print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))
	# save pipeline trained on all data
	pipeline = pipeline.fit(X, y)
	joblib.dump(pipeline, save_name)
	print("model saved to %s" % save_name)

if __name__ == "__main__":
	main()