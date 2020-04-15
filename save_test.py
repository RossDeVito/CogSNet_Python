from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tsc_models import *
from rankers_util import *
from rankers import *
from comparers import *

model_name = 'test_model'

callbacks = [
	EarlyStopping(patience=12, verbose=1, restore_best_weights=True),
	ReduceLROnPlateau(factor=.5, patience=7, verbose=1)
]

model = LSTM1(4)

ranker = TimeSeriesPairwiseRanker(
		TimeSeriesComparer(
			model,
			desc="LSTM1 bs=1024",
			batch_size=1024,
			epochs=200,
			callbacks=callbacks,
			verbose=2,
			validation_split=.1,
			n_workers=20
		),
		bin_size=21,
		other_feat=False,
		text_call_split=True,
		metric='count', # count, val, or both
		verbose=1)

print(ranker)
print(ranker.comparer)
print(ranker.comparer.model.input_shape)
print(ranker.comparer.scaler.get_params())

save_keras_ranker(ranker, path='trained_models', dir_name='test')

loaded_ranker = load_keras_ranker('trained_models/test')

print(loaded_ranker)
print(loaded_ranker.comparer)
print(loaded_ranker.comparer.model.input_shape)
print(loaded_ranker.comparer.scaler.get_params())