from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lfadslite import LFADS
import numpy as np
import os
import tensorflow as tf
import re
import data_funcs as utils

#lfadslite
from helper_funcs import kind_dict, kind_dict_key


## need to implement:
MAX_CKPT_TO_KEEP = 5
MAX_CKPT_TO_KEEP_LVE = 5
CKPT_SAVE_INTERVAL = 5
CSV_LOG = "fitlog"
OUTPUT_FILENAME_STEM = ""
CHECKPOINT_PB_LOAD_NAME = "checkpoint"
CHECKPOINT_NAME = "lfads_vae"
DEVICE = "gpu:0" # "cpu:0", or other gpus, e.g. "gpu:1"
PS_NEXAMPLES_TO_PROCESS = int(1e8) # if larger than number of examples, process all

# L2 weights
L2_GEN_SCALE = 2000.0
L2_CON_SCALE = 0.0
L2_IC_ENC_SCALE = 0.0
L2_CI_ENC_SCALE = 0.0
L2_GEN_2_FACTORS_SCALE = 0.0
L2_CI_ENC_2_CO_IN = 0.0
# inverse-gamma parameter: if using feedforward network, add L2 regularization
L2_FAC_2_RATES_SCALE = 0.0

IC_DIM = 64
FACTORS_DIM = 50
IC_ENC_DIM = 128
IC_ENC_SEG_LEN = 0 # default, non-causal modeling
GEN_DIM = 200
BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
LEARNING_RATE_INIT = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.95
LEARNING_RATE_STOP = 0.00001
LEARNING_RATE_N_TO_COMPARE = 6
N_EPOCHS_EARLY_STOP = 200 # epochs
TARGET_NUM_EPOCHS = 0
DO_RESET_LEARNING_RATE = False

# temporal spike jitter width
TEMPORAL_SPIKE_JITTER_WIDTH = 0

# log transform input
LOG_TRANSFORM_INPUT = False

# temporal shift
TEMPORAL_SHIFT = 0
TEMPORAL_SHIFT_DIST = "normal"
APPLY_TEMPORAL_SHIFT_DURING_POSTERIOR_SAMPLING = False

# flag to only allow training of the encoder (i.e., lock the generator, factors readout, rates readout, controller, etc weights)
DO_TRAIN_ENCODER_ONLY = False

# flag to allow training the readin (alignment) matrices (only used in cases where alignment matrices are used
DO_TRAIN_READIN = True

# lfadslite parameter - sets the dimensionality between ci_enc and controller
CON_CI_ENC_IN_DIM = 10
# lfadslite parameter - sets the dimensionality between factors and controller
CON_FAC_IN_DIM = 10
# lfadslite param -
#     sets whether there is an "input_factors" layer (for multi-session data, or even if you want to reduce the dimensionality of a single session)
IN_FACTORS_DIM = 0


# Calibrated just above the average value for the rnn synthetic data.
MAX_GRAD_NORM = 200.0
CELL_CLIP_VALUE = 5.0
KEEP_PROB = 0.95
# inverse gamma parameter: to determine whethere to use feedforward network output or linear-exponential
FAC_2_RATES_TRANSFORM = 'linexp'
# inverse-gamma parameter: If using feedforward network, set keep_prob ratio for dropout
FF_KEEP_PROB = 0.95
# zi-gamma parameters
GAMMA_PRIOR = 20.0
L2_GAMMA_DISTANCE_SCALE = 1e-4
S_MIN = 0.1

KEEP_RATIO = 1.0
CD_GRAD_PASSTHRU_PROB = 0.0

CV_KEEP_RATIO = 1.0
CV_RAND_SEED = 0.0

OUTPUT_DIST = 'poisson' # 'poisson' or 'gaussian'

DATA_DIR = "/tmp/rnn_synth_data_v1.0/"
DATA_FILENAME_STEM = "chaotic_rnn_inputs_g1p5"
LFADS_SAVE_DIR = "/tmp/lfads_chaotic_rnn_inputs_g1p5/lfadsOut/"
CO_DIM = 1
DO_CAUSAL_CONTROLLER = False
CONTROLLER_INPUT_LAG = 1
CI_ENC_DIM = 128
CON_DIM = 128
EXT_INPUT_DIM = 0

# scale of regularizer on time correlation of inferred inputs
KL_IC_WEIGHT = 1.0
KL_CO_WEIGHT = 1.0
IC_PRIOR_VAR = 0.1
IC_POST_VAR_MIN = 0.0001      # protection from KL blowing up
CO_PRIOR_VAR = 0.1

KL_START_EPOCH = 0
L2_START_EPOCH = 0
KL_INCREASE_EPOCHS = 500
L2_INCREASE_EPOCHS = 500

# params for autoregressive prior for the controller (not used now)
PRIOR_AR_AUTOCORRELATION = 10.0
PRIOR_AR_PROCESS_VAR = 0.1
DO_TRAIN_PRIOR_AR_ATAU = True
DO_TRAIN_PRIOR_AR_NVAR = True

# params for loss scaling/ ADAM optimizer
LOSS_SCALE = 1e4
ADAM_EPSILON = 1e-8
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999


# calculate R^2 if the truth rates are available
DO_CALC_R2 = False

flags = tf.app.flags
flags.DEFINE_string("kind", "train",
                    "Type of model to build {train, \
                    posterior_sample_and_average, posterior_mean, \
                    prior_sample, write_model_params}")
flags.DEFINE_string("output_dist", OUTPUT_DIST,
                    "Type of output distribution, 'poisson' or 'gaussian'")
flags.DEFINE_string("fac_2_rates_transform", FAC_2_RATES_TRANSFORM,
                    "Type of transformation from factors to rates. 'linexp' or 'feedforward'")
flags.DEFINE_boolean("allow_gpu_growth", True,
                     "If true, only allocate amount of memory needed for \
                     Session. Otherwise, use full GPU memory.")

# DATA
flags.DEFINE_string("data_dir", DATA_DIR, "Data for training")
flags.DEFINE_string("data_filename_stem", DATA_FILENAME_STEM,
                    "Filename stem for data dictionaries.")
flags.DEFINE_string("lfads_save_dir", LFADS_SAVE_DIR, "model save dir")
flags.DEFINE_string("checkpoint_pb_load_name", CHECKPOINT_PB_LOAD_NAME,
                    "Name of checkpoint files, use 'checkpoint_lve' for best \
                    error")
flags.DEFINE_string("checkpoint_name", CHECKPOINT_NAME,
                    "Name of checkpoint files (.ckpt appended)")
flags.DEFINE_string("output_filename_stem", OUTPUT_FILENAME_STEM,
                    "Name of output file (postfix will be added)")
flags.DEFINE_string("device", DEVICE,
                    "Which device to use (default: \"gpu:0\", can also be \
                    \"cpu:0\", \"gpu:1\", etc)")
flags.DEFINE_string("csv_log", CSV_LOG,
                    "Name of file to keep running log of fit likelihoods, \
                    etc (.csv appended)")
flags.DEFINE_integer("max_ckpt_to_keep", MAX_CKPT_TO_KEEP,
                 "Max # of checkpoints to keep (rolling)")
flags.DEFINE_integer("ps_nexamples_to_process", PS_NEXAMPLES_TO_PROCESS,
                 "Number of examples to process for posterior sample and \
                 average (not number of samples to average over).")
flags.DEFINE_integer("max_ckpt_to_keep_lve", MAX_CKPT_TO_KEEP_LVE,
                 "Max # of checkpoints to keep for lowest validation error \
                 models (rolling)")

flags.DEFINE_integer("ckpt_save_interval", CKPT_SAVE_INTERVAL,
                 "Number of epochs between saving (non-lve) checkpoints")


# GENERATION

# Note that the dimension of the initial conditions is separated from the
# dimensions of the generator initial conditions (and a linear matrix will
# adapt the shapes if necessary).  This is just another way to control
# complexity.  In all likelihood, setting the ic dims to the size of the
# generator hidden state is just fine.
flags.DEFINE_integer("ic_dim", IC_DIM, "Dimension of h0")
# Setting the dimensions of the factors to something smaller than the data
# dimension is a way to get a reduced dimensionality representation of your
# data.
flags.DEFINE_integer("factors_dim", FACTORS_DIM,
                     "Number of factors from generator")
# The input factors decide the dimensionality of the data that the encoders see
# This is critical for multi-session data, where the encoders must see a consistent input dimensionality across sessions
flags.DEFINE_integer("in_factors_dim", IN_FACTORS_DIM,
                     "Number of 'input factors' (encoders read from these)")
flags.DEFINE_integer("ic_enc_dim", IC_ENC_DIM,
                     "Cell hidden size, encoder of h0")
flags.DEFINE_integer("ic_enc_seg_len", IC_ENC_SEG_LEN,
                     "Segment length passed to IC encoder for causal modeling")


flags.DEFINE_integer("con_ci_enc_in_dim", CON_CI_ENC_IN_DIM,
                     "Dimensionality of (time-varying) input to the controller that comes from the controller input encoder (ci_enc)")
flags.DEFINE_integer("con_fac_in_dim", CON_FAC_IN_DIM,
                     "Dimensionality of (time-varying) input to the controller that comes from the factors")

# Controlling the size of the generator is one way to control complexity of
# the dynamics (there is also l2, which will squeeze out unnecessary
# dynamics also).  The modern deep learning approach is to make these cells
# as large as tolerable (from a waiting perspective), and then regularize
# them to death with drop out or whatever.  I don't know if this is correct
# for the LFADS application or not.
flags.DEFINE_integer("gen_dim", GEN_DIM,
                     "Cell hidden size, generator.")
flags.DEFINE_float("l2_gen_scale", L2_GEN_SCALE,
                   "L2 regularization cost for the generator only.")
flags.DEFINE_float("l2_con_scale", L2_CON_SCALE,
                   "L2 regularization cost for the controller only.")
flags.DEFINE_float("l2_ic_enc_scale", L2_IC_ENC_SCALE,
                   "L2 regularization cost for the generator only.")
flags.DEFINE_float("l2_ci_enc_scale", L2_CI_ENC_SCALE,
                   "L2 regularization cost for the controller only.")
flags.DEFINE_float("l2_gen_2_factors_scale", L2_GEN_2_FACTORS_SCALE,
                   "L2 regularization cost for the generator only.")
flags.DEFINE_float("l2_ci_enc_2_co_in", L2_CI_ENC_2_CO_IN,
                   "L2 regularization cost for the controller only.")
flags.DEFINE_float("l2_fac_2_rates_scale", L2_FAC_2_RATES_SCALE,
                   "L2 regularization cost for the factors 2 rates transform  only.")



# KL DISTRIBUTIONS
# If you don't know what you are donig here, please leave alone, the
# defaults should be fine for most cases, irregardless of other parameters.
#
flags.DEFINE_float("ic_prior_var", IC_PRIOR_VAR,
                   "Minimum variance in posterior h0 codes.")
# If you really want to limit the information from encoder to decoder,
# Increase ic_post_var_min above 0.0.
flags.DEFINE_float("ic_post_var_min", IC_POST_VAR_MIN,
                   "Minimum variance of IC posterior distribution.")
flags.DEFINE_float("co_prior_var", CO_PRIOR_VAR,
                   "Variance of control input prior distribution.")


# Sometimes the task can be sufficiently hard to learn that the
# optimizer takes the 'easy route', and simply minimizes the KL
# divergence, setting it to near zero, and the optimization gets
# stuck.  These two parameters will help avoid that by by getting the
# optimization to 'latch' on to the main optimization, and only
# turning in the regularizers later.
flags.DEFINE_integer("kl_start_epoch", KL_START_EPOCH,
                     "Start increasing weight after this many steps.")
# training passes, not epochs, increase by 0.5 every kl_increase_epochs
flags.DEFINE_integer("kl_increase_epochs", KL_INCREASE_EPOCHS,
                     "Increase weight of kl cost to avoid local minimum.")
# Same story for l2 regularizer.  One wants a simple generator, for scientific
# reasons, but not at the expense of hosing the optimization.
flags.DEFINE_integer("l2_start_epoch", L2_START_EPOCH,
                     "Start increasing l2 weight after this many steps.")
flags.DEFINE_integer("l2_increase_epochs", L2_INCREASE_EPOCHS,
                     "Increase weight of l2 cost to avoid local minimum.")


# EXTERNAL INPUTS
flags.DEFINE_integer("ext_input_dim", EXT_INPUT_DIM,
    "Dimension of external inputs if any.")

# CONTROLLER
# This parameter critically controls whether or not there is a controller
# (along with controller encoders placed into the LFADS graph.  If CO_DIM >
# 1, that means there is a 1 dimensional controller outputs, if equal to 0,
# then no controller.
flags.DEFINE_integer("co_dim", CO_DIM,
    "Number of control net outputs (>0 builds that graph).")

flags.DEFINE_float("prior_ar_atau",  PRIOR_AR_AUTOCORRELATION,
                   "Initial autocorrelation of AR(1) priors.")
flags.DEFINE_float("prior_ar_nvar", PRIOR_AR_PROCESS_VAR,
                   "Initial noise variance for AR(1) priors.")
flags.DEFINE_boolean("do_train_prior_ar_atau", DO_TRAIN_PRIOR_AR_ATAU,
                     "Is the value for atau an init, or the constant value?")
flags.DEFINE_boolean("do_train_prior_ar_nvar", DO_TRAIN_PRIOR_AR_NVAR,
                     "Is the value for noise variance an init, or the constant \
                     value?")



# The controller will be more powerful if it can see the encoding of the entire
# trial.  However, this allows the controller to create inferred inputs that are
# acausal with respect to the actual data generation process.  E.g. the data
# generator could have an input at time t, but the controller, after seeing the
# entirety of the trial could infer that the input is coming a little before
# time t, because there are no restrictions on the data the controller sees.
# One can force the controller to be causal (with respect to perturbations in
# the data generator) so that it only sees forward encodings of the data at time
# t that originate at times before or at time t.  One can also control the data
# the controller sees by using an input lag (forward encoding at time [t-tlag]
# for controller input at time t.  The same can be done in the reverse direction
# (controller input at time t from reverse encoding at time [t+tlag], in the
# case of an acausal controller).  Setting this lag > 0 (even lag=1) can be a
# powerful way of avoiding very spiky decodes. Finally, one can manually control
# whether the factors at time t-1 are fed to the controller at time t.
#
flags.DEFINE_integer("ci_enc_dim", CI_ENC_DIM,
                     "Cell hidden size, encoder of control inputs")
flags.DEFINE_integer("con_dim", CON_DIM,
                     "Cell hidden size, controller")
flags.DEFINE_boolean("do_causal_controller",
                     DO_CAUSAL_CONTROLLER,
                     "Restrict the controller create only causal inferred \
                     inputs?")
flags.DEFINE_integer("controller_input_lag", CONTROLLER_INPUT_LAG, 
                     "Time lag on the encoding to controller t-lag for \
                     forward, t+lag for reverse.")


# OPTIMIZATION
flags.DEFINE_integer("batch_size", BATCH_SIZE,
                     "Batch size to use during training.")
flags.DEFINE_integer("valid_batch_size", None,
                     "Batch size to use during validation.")
flags.DEFINE_float("learning_rate_init", LEARNING_RATE_INIT,
                   "Learning rate initial value")
flags.DEFINE_float("learning_rate_decay_factor", LEARNING_RATE_DECAY_FACTOR,
                   "Learning rate decay, decay by this fraction every so \
                   often.")
flags.DEFINE_float("learning_rate_stop", LEARNING_RATE_STOP,
                   "The lr is adaptively reduced, stop training at this value.")
# Rather put the learning rate on an exponentially decreasiong schedule,
# the current algorithm pays attention to the learning rate, and if it
# isn't regularly decreasing, it will decrease the learning rate.  So far,
# it works fine, though it is not perfect.
flags.DEFINE_integer("learning_rate_n_to_compare", LEARNING_RATE_N_TO_COMPARE,
                     "Number of previous costs current cost has to be worse \
                     than, to lower learning rate.")
flags.DEFINE_integer("n_epochs_early_stop", N_EPOCHS_EARLY_STOP,
                     "Number of previous costs current cost has to be worse \
                     than, to lower learning rate.")
flags.DEFINE_integer("target_num_epochs", TARGET_NUM_EPOCHS,
                     "Number of epochs to run before stopping.")

flags.DEFINE_boolean("do_reset_learning_rate", DO_RESET_LEARNING_RATE,
                     "Reset the learning rate to initial value.")


# This sets a value, above which, the gradients will be clipped.  This hp
# is extremely useful to avoid an infrequent, but highly pathological
# problem whereby the gradient is so large that it destroys the
# optimziation by setting parameters too large, leading to a vicious cycle
# that ends in NaNs.  If it's too large, it's useless, if it's too small,
# it essentially becomes the learning rate.  It's pretty insensitive, though.
flags.DEFINE_float("max_grad_norm", MAX_GRAD_NORM,
                   "Max norm of gradient before clipping.")
# If your optimizations start "NaN-ing out", reduce this value so that
# the values of the network don't grow out of control.  Typically, once
# this parameter is set to a reasonable value, one stops having numerical
# problems.
flags.DEFINE_float("cell_clip_value", CELL_CLIP_VALUE,
                   "Max value recurrent cell can take before being clipped.")

# This flag is used for an experiment where one wants to know if the dynamics
# learned by the generator generalize across conditions. In that case, you might
# train up a model on one set of data, and then only further train the encoder on 
# another set of data (the conditions to be tested) so that the model is forced
# to use the same dynamics to describe that data.
# If you don't care about that particular experiment, this flag should always be
# false.
flags.DEFINE_boolean("do_train_encoder_only", DO_TRAIN_ENCODER_ONLY,
                     "Train only the encoder weights.")


# for multi-session "stitching" models, the per-session readin matrices map from
# neurons to input factors which are fed into the shared encoder. These are
# initialized by alignment_matrix_cxf and alignment_bias_c in the input .h5
# files. They can be fixed or made trainable.
flags.DEFINE_boolean("do_train_readin", DO_TRAIN_READIN, "Whether to train the \
                     readin matrices and bias vectors. False leaves them fixed \
                     at their initial values specified by the alignment \
                     matrices and vectors.")



# OVERFITTING
# Dropout is done on the input data, on controller inputs (from
# encoder), on outputs from generator to factors.
flags.DEFINE_float("keep_prob", KEEP_PROB, "Dropout keep probability.")
flags.DEFINE_float("ff_keep_prob", FF_KEEP_PROB, "Dropout keep probability on the feedforward network between factors -> rates (if `feedforward` option is used for fac_2_rates_transform)")

# COORDINATED DROPOUT
flags.DEFINE_float("keep_ratio", KEEP_RATIO, "Coordinated Dropout input keep probability.")
flags.DEFINE_float("cd_grad_passthru_prob", CD_GRAD_PASSTHRU_PROB, "Probability of passing through gradients in coordinated dropout.")

# CROSS-VALIDATION
flags.DEFINE_float("cv_keep_ratio", CV_KEEP_RATIO, "Cross-validation keep probability.")

# CROSS-VALIDATION RANDOM SEED
flags.DEFINE_float("cv_rand_seed", CV_RAND_SEED, "Random seed for held-out cross-validation sample mask.")

# TEMPORAL SPIKE JITTER 
# It appears that the system will happily fit spikes (blessing or
# curse, depending).  You may not want this.  Jittering the spikes a
# bit will help (-/+ bin size, as specified here).
flags.DEFINE_integer("temporal_spike_jitter_width",
                     TEMPORAL_SPIKE_JITTER_WIDTH,
                     "Shuffle spikes around this window.")
# LOG TRANSFORM INPUT
flags.DEFINE_boolean("log_transform_input",
                     LOG_TRANSFORM_INPUT,
                    "Flag toggling whether to log transform input to the encoder")


# TEMPORAL SHIFT
# In the case of continous data, such as EMG, it seems LFADS will
# over-couple channels to choose model high frequency events
# Temporal shifting channels is an attempt to shift the data randomly
# by n samples for each channel, thus breaking any directly correlated
# events. This strategy thus allows us to prevent the system from 
# modeling instantaneous correlations and instead focus on modeling
# more robust correlations in the data. 
flags.DEFINE_integer("temporal_shift",
                     TEMPORAL_SHIFT,
                     "Randomly shift input data channels by +/- n samples based on value.")
flags.DEFINE_string("temporal_shift_dist", TEMPORAL_SHIFT_DIST,
                    "Define dist. to randomly sample shifts for temporal shift operation. 'normal' or 'uniform'")
flags.DEFINE_boolean("apply_temporal_shift_during_posterior_sampling",
                     APPLY_TEMPORAL_SHIFT_DURING_POSTERIOR_SAMPLING,
                    "Flag toggling whether to apply temporal shift during posterior sampling")
# ZIG related parameters
flags.DEFINE_float("gamma_prior", GAMMA_PRIOR, "prior of scaling of sigmoid nonlinearity")
flags.DEFINE_float("l2_gamma_distance_scale", L2_GAMMA_DISTANCE_SCALE, "strength of penalty applied for deviating from gamma prior")
flags.DEFINE_float("s_min", S_MIN, "minimal event size for ZIG model")
# UNDERFITTING
# If the primary task of LFADS is "filtering" of data and not
# generation, then it is possible that the KL penalty is too strong.
# Empirically, we have found this to be the case.  So we add a
# hyperparameter in front of the the two KL terms (one for the initial
# conditions to the generator, the other for the controller outputs).
# You should always think of the the default values as 1.0, and that
# leads to a standard VAE formulation whereby the numbers that are
# optimized are a lower-bound on the log-likelihood of the data. When
# these 2 HPs deviate from 1.0, one cannot make any statement about
# what those LL lower bounds mean anymore, and they cannot be compared
# (AFAIK).
flags.DEFINE_float("kl_ic_weight", KL_IC_WEIGHT,
                   "Strength of KL weight on initial conditions KL penatly.")
flags.DEFINE_float("kl_co_weight", KL_CO_WEIGHT,
                   "Strength of KL weight on controller output KL penalty.")

# LOSS SCALING/ADAM OPTIMIZER PRESETS
flags.DEFINE_float("loss_scale", LOSS_SCALE,
                   "Scaling of loss.")
flags.DEFINE_float("adam_epsilon", ADAM_EPSILON,
                   "Epsilon parameter of ADAM optimizer.")
flags.DEFINE_float("adam_beta1", ADAM_BETA1,
                   "Beta1 parameter of ADAM optimizer.")
flags.DEFINE_float("adam_beta2", ADAM_BETA2,
                   "Beta2 parameter of ADAM optimizer.")

flags.DEFINE_boolean("do_calc_r2", True,
                     "Calculate R^2 is the truth rates are available.")


FLAGS = flags.FLAGS



def build_model(hps, datasets=None):
  """Builds a model from either random initialization, or saved parameters.

  Args:
    hps: The hyper parameters for the model.
    datasets: The datasets structure (see top of lfads.py).

  Returns:
    an LFADS model.
  """

  with tf.variable_scope("LFADS", reuse=None):
    model = LFADS(hps, datasets=datasets)

  if not os.path.exists(hps.lfads_save_dir):
    print("Save directory %s does not exist, creating it." % hps.lfads_save_dir)
    os.makedirs(hps.lfads_save_dir)

  cp_pb_ln = hps.checkpoint_pb_load_name
  cp_pb_ln = 'checkpoint' if cp_pb_ln == "" else cp_pb_ln
  if cp_pb_ln == 'checkpoint':
    print("Loading latest training checkpoint in: ", hps.lfads_save_dir)
    saver = model.seso_saver
  elif cp_pb_ln == 'checkpoint_lve':
    print("Loading lowest validation checkpoint in: ", hps.lfads_save_dir)
    saver = model.lve_saver
  else:
    print("Loading checkpoint: ", cp_pb_ln, ", in: ", hps.lfads_save_dir)
    saver = model.seso_saver

  ckpt = tf.train.get_checkpoint_state(hps.lfads_save_dir,
                                       latest_filename=cp_pb_ln)

  session = tf.get_default_session()
  print("ckpt: ", ckpt)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    if hps.kind in [kind_dict("posterior_sample_and_average"),
                    kind_dict("prior_sample"),
                    kind_dict("write_model_params")]:
      print("Possible error!!! You are running ", kind_dict_key(hps.kind), " on a newly \
      initialized model!")
      print("Are you sure you sure ", hps.lfads_save_dir, " exists?")

    tf.global_variables_initializer().run()

  if ckpt:
    train_step_str = re.search('-[0-9]+$', ckpt.model_checkpoint_path).group()
  else:
    train_step_str = '-0'

  fname = 'hyperparameters' + train_step_str + '.txt'
  hp_fname = os.path.join(hps.lfads_save_dir, fname)
  hps_for_saving = jsonify_dict(hps)
  utils.write_data(hp_fname, hps_for_saving, use_json=True)



  return model


def jsonify_dict(d):
  """Turns python booleans into strings so hps dict can be written in json.
  Creates a shallow-copied dictionary first, then accomplishes string
  conversion.

  Args: 
    d: hyperparameter dictionary

  Returns: hyperparameter dictionary with bool's as strings
  """

  d2 = d.copy()   # shallow copy is fine by assumption of d being shallow
  def jsonify_bool(boolean_value):
    if boolean_value:
      return "true"
    else:
      return "false"

  for key in d2.keys():
    if isinstance(d2[key], bool):
      d2[key] = jsonify_bool(d2[key])
  return d2


def build_hyperparameter_dict(flags):
  """Simple script for saving hyper parameters.  Under the hood the
  flags structure isn't a dictionary, so it has to be simplified since we
  want to be able to view file as text.

  Args:
    flags: From tf.app.flags

  Returns:
    dictionary of hyper parameters (ignoring other flag types).
  """
  d = {}
  # Data
# CP - in this model we use enumerations for the 'kind'
  #  d['kind'] = flags.kind
  d['kind'] = kind_dict(flags.kind)
  
  d['output_dist'] = flags.output_dist
  d['fac_2_rates_transform'] = flags.fac_2_rates_transform
    
  d['data_dir'] = flags.data_dir
  d['lfads_save_dir'] = flags.lfads_save_dir
  d['checkpoint_pb_load_name'] = flags.checkpoint_pb_load_name
  d['checkpoint_name'] = flags.checkpoint_name
  d['output_filename_stem'] = flags.output_filename_stem
  d['max_ckpt_to_keep'] = flags.max_ckpt_to_keep
  d['max_ckpt_to_keep_lve'] = flags.max_ckpt_to_keep_lve
  d['ckpt_save_interval'] = flags.ckpt_save_interval
  d['ps_nexamples_to_process'] = flags.ps_nexamples_to_process
  d['data_filename_stem'] = flags.data_filename_stem
  d['device'] = flags.device
  d['csv_log'] = flags.csv_log
  # Generation
  d['ic_dim'] = flags.ic_dim
  d['factors_dim'] = flags.factors_dim
  d['ic_enc_dim'] = flags.ic_enc_dim
  d['ic_enc_seg_len'] = flags.ic_enc_seg_len
  d['gen_dim'] = flags.gen_dim

  #lfadslite
  d['con_ci_enc_in_dim'] = flags.con_ci_enc_in_dim
  d['con_fac_in_dim'] = flags.con_fac_in_dim
  d['in_factors_dim'] = flags.in_factors_dim

  # KL distributions
  d['ic_prior_var'] = flags.ic_prior_var
  d['ic_post_var_min'] = flags.ic_post_var_min
  d['co_prior_var'] = flags.co_prior_var

  # Controller
  d['do_causal_controller'] = flags.do_causal_controller
  d['controller_input_lag'] = flags.controller_input_lag
  d['prior_ar_atau'] = flags.prior_ar_atau
  d['prior_ar_nvar'] = flags.prior_ar_nvar
  d['do_train_prior_ar_atau'] = flags.do_train_prior_ar_atau
  d['do_train_prior_ar_nvar'] = flags.do_train_prior_ar_nvar

#  d['do_feed_factors_to_controller'] = flags.do_feed_factors_to_controller
  d['co_dim'] = flags.co_dim
  d['ext_input_dim'] = flags.ext_input_dim
  d['ci_enc_dim'] = flags.ci_enc_dim
  d['con_dim'] = flags.con_dim
  # Optimization
  d['batch_size'] = flags.batch_size
  d['valid_batch_size'] = flags.batch_size if flags.valid_batch_size is None else flags.valid_batch_size
  d['learning_rate_init'] = flags.learning_rate_init
  d['learning_rate_decay_factor'] = flags.learning_rate_decay_factor
  d['learning_rate_stop'] = flags.learning_rate_stop
  d['learning_rate_n_to_compare'] = flags.learning_rate_n_to_compare
  d['n_epochs_early_stop'] = flags.n_epochs_early_stop
  d['target_num_epochs'] = flags.target_num_epochs
  d['do_reset_learning_rate'] = flags.do_reset_learning_rate
  d['max_grad_norm'] = flags.max_grad_norm
  d['cell_clip_value'] = flags.cell_clip_value

  # training options
  d['do_train_encoder_only']  = flags.do_train_encoder_only
  d['do_train_readin']  = flags.do_train_readin

  d['log_transform_input'] = flags.log_transform_input
  
  # Overfitting
  d['keep_prob'] = flags.keep_prob
  d['ff_keep_prob'] = flags.ff_keep_prob

  
  d['temporal_shift'] = flags.temporal_shift
  d['temporal_shift_dist'] = flags.temporal_shift_dist
  d['apply_temporal_shift_during_posterior_sampling'] = flags.apply_temporal_shift_during_posterior_sampling
  d['temporal_spike_jitter_width'] = flags.temporal_spike_jitter_width
  d['keep_ratio'] = flags.keep_ratio
  d['cd_grad_passthru_prob'] = flags.cd_grad_passthru_prob
  d['cv_keep_ratio'] = flags.cv_keep_ratio
  d['cv_rand_seed'] = flags.cv_rand_seed
  d['gamma_prior'] = flags.gamma_prior
  d['s_min'] = flags.s_min

  d['l2_gen_scale'] = flags.l2_gen_scale
  d['l2_con_scale'] = flags.l2_con_scale
  d['l2_ic_enc_scale'] = flags.l2_ic_enc_scale
  d['l2_ci_enc_scale'] = flags.l2_ci_enc_scale
  d['l2_gen_2_factors_scale'] = flags.l2_gen_2_factors_scale
  d['l2_ci_enc_2_co_in'] = flags.l2_ci_enc_2_co_in
  d['l2_fac_2_rates_scale'] = flags.l2_fac_2_rates_scale
  d['l2_gamma_distance_scale'] = flags.l2_gamma_distance_scale
  
  # Underfitting
  d['kl_ic_weight'] = flags.kl_ic_weight
  d['kl_co_weight'] = flags.kl_co_weight

  d['kl_start_epoch'] = flags.kl_start_epoch
  d['kl_increase_epochs'] = flags.kl_increase_epochs
  d['l2_start_epoch'] = flags.l2_start_epoch
  d['l2_increase_epochs'] = flags.l2_increase_epochs

  # Loss scaling/Adam Optimizer Presets
  d['loss_scale'] = flags.loss_scale
  d['adam_epsilon'] = flags.adam_epsilon
  d['beta1'] = flags.adam_beta1
  d['beta2'] = flags.adam_beta2

  d['do_calc_r2'] = flags.do_calc_r2
  
  return d

class hps_dict_to_obj(dict):
  """Helper class allowing us to access hps dictionary more easily."""

  def __getattr__(self, key):
    if key in self:
      return self[key]
    else:
      assert False, ("%s does not exist." % key)
  def __setattr__(self, key, value):
    self[key] = value



def train(hps, datasets):
  """Train the LFADS model.

  Args:
    hps: The dictionary of hyperparameters.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
  """
  model = build_model(hps, datasets=datasets)
  if hps.do_reset_learning_rate:
    sess = tf.get_default_session()
    sess.run(model.learning_rate.initializer)

  model.train_model(datasets)


def write_model_runs(hps, datasets, output_fname=None):
  """Run the model on the data in data_dict, and save the computed values.

  LFADS generates a number of outputs for each examples, and these are all
  saved.  They are:
    The mean and variance of the prior of g0.
    The mean and variance of approximate posterior of g0.
    The control inputs (if enabled)
    The initial conditions, g0, for all examples.
    The generator states for all time.
    The factors for all time.
    The rates for all time.

  Args:
    hps: The dictionary of hyperparameters.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
    output_fname (optional): output filename stem to write the model runs.
  """
  model = build_model(hps, datasets=datasets)
  model.write_model_runs(datasets, output_fname)


def write_model_samples(hps, datasets, dataset_name=None, output_fname=None):
  """Use the prior distribution to generate samples from the model.
  Generates batch_size number of samples (set through FLAGS).

  LFADS generates a number of outputs for each examples, and these are all
  saved.  They are:
    The mean and variance of the prior of g0.
    The control inputs (if enabled)
    The initial conditions, g0, for all examples.
    The generator states for all time.
    The factors for all time.
    The output distribution parameters (e.g. rates) for all time.

  Args:
    hps: The dictionary of hyperparameters.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
    dataset_name: The name of the dataset to grab the factors -> rates
      alignment matrices from. Only a concern with models trained on
      multi-session data. By default, uses the first dataset in the data dict.
    output_fname: The name prefix of the file in which to save the generated
      samples.
  """

  # get kind_dict keys
  kind_str = kind_dict_key(hps.kind)
  
  if not output_fname:
    output_fname = "model_runs_" + kind_str
  else:
    output_fname = output_fname + "model_runs_" + kind_str
  if not dataset_name:
    dataset_name = datasets.keys()[0]
  else:
    if dataset_name not in datasets.keys():
      raise ValueError("Invalid dataset name '%s'."%(dataset_name))
  model = build_model(hps, datasets=datasets)
  model.write_model_samples(dataset_name, output_fname)


def write_model_parameters(hps, output_fname=None, datasets=None):
  """Save all the model parameters

  Save all the parameters to hps.lfads_save_dir.

  Args:
    hps: The dictionary of hyperparameters.
    output_fname: The prefix of the file in which to save the generated
      samples.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
  """
  if not output_fname:
    output_fname = "model_params"
  else:
    output_fname = output_fname + "_model_params"
  fname = os.path.join(hps.lfads_save_dir, output_fname)
  print("Writing model parameters to: ", fname)
  # save the optimizer params as well
  model = build_model(hps, datasets=datasets) 
  model_params = model.eval_model_parameters(use_nested=False,
                                             include_strs="LFADS")
  utils.write_data(fname, model_params, compression=None)
  print("Done.")


def clean_data_dict(data_dict):
  """Add some key/value pairs to the data dict, if they are missing.
  Args:
    data_dict - dictionary containing data for LFADS
  Returns:
    data_dict with some keys filled in, if they are absent.
  """

  keys = ['train_truth', 'train_ext_input', 'valid_data',
          'valid_truth', 'valid_ext_input', 'valid_train']
  for k in keys:
    if k not in data_dict:
      data_dict[k] = None

  return data_dict

#
# def load_datasets(data_dir, data_filename_stem, hps):
#   """Load the datasets from a specified directory.
#
#   Example files look like
#     >data_dir/my_dataset_first_day
#     >data_dir/my_dataset_second_day
#
#   If my_dataset (filename) stem is in the directory, the read routine will try
#   and load it.  The datasets dictionary will then look like
#   dataset['first_day'] -> (first day data dictionary)
#   dataset['second_day'] -> (first day data dictionary)
#
#   Args:
#     data_dir: The directory from which to load the datasets.
#     data_filename_stem: The stem of the filename for the datasets.
#
#   Returns:
#     datasets: a dataset dictionary, with one name->data dictionary pair for
#     each dataset file.
#   """
#   print("Reading data from ", data_dir)
#   datasets = utils.read_datasets(data_dir, data_filename_stem)
#   for k, data_dict in datasets.items():
#     datasets[k] = clean_data_dict(data_dict)
#
#     train_total_size = len(data_dict['train_data'])
#     if train_total_size == 0:
#       print("Did not load training set.")
#     else:
#       print("Found training set with number examples: ", train_total_size)
#
#     valid_total_size = len(data_dict['valid_data'])
#     if valid_total_size == 0:
#       print("Did not load validation set.")
#     else:
#       print("Found validation set with number examples: ", valid_total_size)
#
#   return datasets


def main(_):
  """Get this whole shindig off the ground."""
  d = build_hyperparameter_dict(FLAGS)
  hps = hps_dict_to_obj(d)    # hyper parameters

  # Read the data, if necessary.
  train_set = valid_set = None
  if hps.kind in [kind_dict("train"),
                  kind_dict("posterior_sample_and_average"),
                  kind_dict("prior_sample"),
                  kind_dict("write_model_params")]:
    datasets = utils.load_datasets(hps.data_dir, hps.data_filename_stem, hps)
  else:
    raise ValueError('Kind {} is not supported.'.format(kind))

  # infer the dataset names and dataset dimensions from the loaded files
  hps.dataset_names = []
  hps.dataset_dims = {}
  for key in datasets:
    hps.dataset_names.append(key)
    hps.dataset_dims[key] = datasets[key]['data_dim']

  # also store down the dimensionality of the data
  # - just pull from one set, required to be same for all sets
  hps.num_steps = datasets.values()[0]['num_steps']
  hps.ndatasets = len(hps.dataset_names)


  # we require that in_factors_dim is set if multi-session is used
  if hps.in_factors_dim ==0 and hps.ndatasets > 1:
      raise SyntaxError('For multi-session data, must define in_factors_dim (this is specific to lfadslite)')
    

  
  # CP: not implemented in lfadslite
#  if hps.num_steps_for_gen_ic > hps.num_steps:
#    hps.num_steps_for_gen_ic = hps.num_steps

  tf.reset_default_graph();
  # Build and run the model, for varying purposes.
  config = tf.ConfigProto(allow_soft_placement=True,
                          log_device_placement=False)
  if FLAGS.allow_gpu_growth:
    config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  with sess.as_default():
    with tf.device(hps.device):
      if hps.kind == kind_dict("train"):
        train(hps, datasets)
      elif hps.kind == kind_dict("posterior_sample_and_average"):
        write_model_runs(hps, datasets, hps.output_filename_stem)
      elif hps.kind == kind_dict("prior_sample"):
        write_model_samples(hps, datasets, hps.output_filename_stem)
      elif hps.kind == kind_dict("write_model_params"):
        write_model_parameters(hps, hps.output_filename_stem, datasets)
      else:
        assert False, ("Kind %s is not implemented. " % kind)


if __name__ == "__main__":
    tf.app.run()
 
