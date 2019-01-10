# Kaggle PLAsTiCC Astronomical Classification - 70th place solution

**Challenge summary:** multiclass classification of astronomical objects from their lightcurve timeseries data which is often sparse, unequally-spaced and noisy. Additionaly, the test set distribution is very different from the training set, containing much fainter and more distant objects, as well as a previously unseen object class (ie. not present in training)

Please refer to https://www.kaggle.com/michaelapers/the-plasticc-astronomy-starter-kit for a more comprehensive intro by the competition organizers

## 1. Solution overview

Stack of 9 vanilla MLPs (with varying depth/width and varying feature sets) and 2 LGBMs, with a very shallow LGBM as meta learner

Best ensemble score:
- ~0.92 public, ~0.96 private

Best single-model scores:
- MLP ~1.21 public, ~1.16 private
- LGBM ~0.96 public, ~1.01 private

**Quick remark:** the 0.04 public-private drop hints at some public LB overfitting. This was probably due to excessive LGBM feature tuning, as the MLPs have no trouble generalizing to the private dataset

**Final feature set**
- Time-independent features applied to both raw and distance modulus corrected flux values such as min, max, median, std, skew, kurtosis
- Cross-band relative contributions of a subset of the above features eg. given a curve with a raw flux max of 1500 in band 0, 500 in band 1 and 0 in bands 2 to 6, its cross-band contribution vector would be [0.75 0.25 0 0 0 0]
- Time-dependent features:
  - Linear regression slope and intercept on detected points per curve and band from time of maximum light, weighted by flux error
  - Decay rate for an exponential curve fitted via LMSE, from time of maximum light as above
- Detection-related features:
  - Delta time from first to last detection (see [this discussion](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538))
  - Average detection streak duration, combined across bands
  
**Additional tricks**
- Increased sample weight for samples with higher redshift based on empirical EDA (exploratory data analysis), since we knew that the test set contained fainter objects (see covariate_shift notebook in notebooks)

**Performace-related remarks**
- Multiprocessing was used whenever possible, on chunks of the >16GB test data
- Feature computations used numpy on top of precomputed data aggregations, instead of pandas
- Feature sets were computed in a modular fashion and saved to avoid recomputing

**Other**
- 5-fold CV (cross validation)
- Minimal hyperparam tuning to avoid overfitting, since test set differs significatly from training set

## 2. What was tried but didn't work (non-exhaustive)

- Dataset augmentation per lightcurve - interpolating between consecutive points and adding gaussian noise
- Complex features from cesium package - see http://cesium-ml.org/docs/feature_table.html
- Features related to lightcurve second derivatives
- Features A0, t0, Trise, Tfall resulting from fitting equation (1) in https://arxiv.org/pdf/1008.1024.pdf to lightcurve data
- Features related to time offsets between times of maximum light
- Features based on normalized flux values
- Features that aimed to capture supernova "second maxima" or "shoulders", especially in higher bands
- Features aiming to measure signal noise or "spikeness"
- RMSE to several supernova models created from training data
- Lomb-scargle periodic features using GPU acceleration - see https://cuvarbase.readthedocs.io/en/devel/index.html
- Unsupervised novelty detection using a One Class SVM
- Extensive LB probing and solving for class frequencies and redshifts
- All kinds of postprocessing adjustments to class probability histograms

## 3. In this repo

Important items marked in bold. Note that not everything was pushed to the repo due to size constraints

- confusion_matrices: multiclass confusion matrices
- data: raw competition data (train and test data (not in repo due to size), metadata, data_note, sample submission), lightcurve data chunked, precomputed and aggregated in different formats, augmented datasets, among other misc. data pieces
- edas: random EDA pictures
- level_1_preds: predictions from the first stacking layer (both for test set and OOF (out-of-fold))
- level_2_preds: predictions from the second stacking layer (ie. final predictions)
- level_3_preds: predictions from the 3rd stacking layer (unused in this case due to overfitting)
- models: MLP models for each fold (LGBM models not saved since they're fast (<30secs) to train)
- notebooks: EDAs, idea prototypes, random dataset analysis, visualizations
  - cesium_intro: first try with the cesium package
  - chunk_study: first tries on how to correctly split data into chunks
  - class_99_study: several experiments on how to predict the hidden class 99
  - covariate_shift: experiments on the different redshift distributions between train and test, aiming to derive an empirical formula to reweight training set samples
  - dataset_augmentation: visualizing augmented curves
  - novelty_detection: first draft for novelty detection (one-class svm)
  - postprocess: first draft for postprocessing class probability histograms
  - sn_confusion_study: EDA searching for differences between different supernovae types
  - sn_fit: experiments on supernovae model fitting
  - sub_analysis: visualization of submissions generated with LGBM models
  - supernovae_exploration: further visualization of supernovae classes. See sn_confusion_study above
  - test_redshift: analysis on test set redshift
- subs: store submissions here
- subs_freq_probe: LB probing-related subs, class frequencies
- subs_rs_probe: LB probing-related subs, redshift distribution
- utils: utility scripts
  - calculate_freq_probe_results.py: (LB probing)
  - calculate_rs_probe_results.py: (LB probing)
  - create_freq_probe_subs.py: (LB probing)
  - create_rs_probe_subs.py: (LB probing)
  - file_chunker.py: used to split large test set into smaller chunks
  - misc_utils.py: contains a myriad of utility functions
  - preprocess.py: apply basic preprocessing to competition raw data
  - weight_system.py: small script I shared with community at https://www.kaggle.com/ganfear/calculate-exact-class-weights to calculate the exact class weights for this challenge
- augment_dataset.py: code for dataset augmentation
- fast_sn_feat_train.py: prototype to test usage of feats derived from SN fits of model (1) in https://arxiv.org/pdf/1008.1024.pdf
- featgen_gpu_LS.py: generate lomb scargle features with GPU acceleration
- **feature_engineering.py: modular generation of feature sets, using multiprocessing on chunked data**
- knn.py: KNN model class. Unused in ensemble due to high training time
- **lgbm.py: LGBM model class**
- **nn.py: MLP model class**
- novelty.py: one-class svm prototype for novelty detection of "unseen" class 99 objects
- postprocess.py: postprocessing adjustment of class probability histograms
- sn_fit.py: fit model (1) in https://arxiv.org/pdf/1008.1024.pdf to lightcurve data
- **stack_level_1.py: train first stack level models**
- **stack_level_2.py: train 2nd stack level models**
- stack_level_3.py: train 3rd stack level models
- svm.py: SVM model class. Unused in ensemble due to high training time

## 4. How to generate submissions

1. Run utils/file_chunker.py on train and test sets (downloaded from https://www.kaggle.com/c/PLAsTiCC-2018/data) to split them into smaller chunks
2. Run for eg. store_chunk_lightcurves_cesium from utils/misc_utils.py on chunked lightcurves from 1. to convert them to a cesium-like format
3. Run feature_engineering.py to generate the desired features, for both train and test sets
4. Run stack_level_1.py to train the desired models on features calculated in 3.
5. (optional) Run stack_level_2.py to train a meta-learner on output from 1st level models in 4.
