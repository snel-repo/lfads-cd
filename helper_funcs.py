from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import os
import sys
import warnings
import errno
import pdb

def kind_dict_definition():
    # used in the graph's keep probability
    return {
        'train': 1,
        'posterior_sample_and_average': 2,
        'posterior_mean': 3,
        'prior_sample': 4,
        'write_model_params': 5,
    }


def kind_dict(kind_str):
    # used in the graph's keep probability
    kd = kind_dict_definition()
    return kd[kind_str]


def kind_dict_key(kind_number):
    # get the key for a certain ind
    kd = kind_dict_definition()
    for key, val in kd.iteritems():
        if val == kind_number:
            return key


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def printer(data):
    # prints on the same line
    sys.stdout.write("\r\x1b[K" + data.__str__())
    sys.stdout.flush()

def init_linear_transform(in_size, out_size, name=None, collections=None, mat_init_value=None, bias_init_value=None, normalized=False, do_bias=True):
    # generic function (we use linear transforms in a lot of places)
    # initialize the weights of the linear transformation based on the size of the inputs

    # initialze with a random distribuion
    if mat_init_value is None:
        stddev = 1.0 / np.sqrt(float(in_size))
        mat_init = tf.random_normal_initializer(0.0, stddev, dtype=tf.float32)
        #mat_init = tf.contrib.layers.xavier_initializer()
        vshape = [in_size, out_size]
    else:
        mat_init = tf.constant_initializer(mat_init_value)
        vshape = [in_size, out_size]

    # weight matrix
    w_collections = [tf.GraphKeys.GLOBAL_VARIABLES, "norm-variables"]
    if collections:
        w_collections += collections
    wname = (name + "/W") if name else "/W"
    w = tf.get_variable(wname, vshape, initializer=mat_init,
                        dtype=tf.float32, collections=w_collections)
    if normalized:
        w = tf.nn.l2_normalize(w, axis=0)

    # biases
    bname = (name + "/b") if name else "/b"

    if do_bias:
        if bias_init_value is None:
            b_init = tf.zeros_initializer()
            vshape = [1, out_size]
        else:
            b_init = tf.constant_initializer(bias_init_value)
            vshape = None
        b = tf.get_variable(bname, vshape,
                            initializer=b_init,
                            dtype=tf.float32)
    else:
        b = tf.zeros([1, out_size],
                      name = bname,
                      dtype=tf.float32)
    return (w, b)


def linear(x, out_size, name, collections=None, mat_init_value=None, bias_init_value=None, normalized=False, do_bias=True):
    # generic function (we use linear transforms in a lot of places)
    # initialize the weights of the linear transformation based on the size of the inputs
    in_size = int(x.get_shape()[1])
    W, b = init_linear_transform(in_size, out_size, name=name, collections=collections, mat_init_value=mat_init_value,
                                 bias_init_value=bias_init_value, normalized=normalized, do_bias=do_bias)
    return tf.matmul(x, W) + b


def ListOfRandomBatches(num_trials, batch_size):
    if num_trials <= batch_size:
        warnings.warn("Your batch size is bigger than num_trials! Using single batch ...")
        return [np.random.permutation(range(num_trials))]

    random_order = np.random.permutation(range(num_trials))
    even_num_of_batches = int(np.floor(num_trials / batch_size))
    trials_to_keep = even_num_of_batches * batch_size
    # if num_trials % batch_size != 0:
    #    print("Warning: throwing out %i trials per epoch" % (num_trials-trials_to_keep) )
    random_order = random_order[0:(trials_to_keep)]
    batches = [random_order[i:i + batch_size] for i in range(0, len(random_order), batch_size)]
    return batches


class Gaussian(object):
    """Base class for Gaussian distribution classes."""

    @property
    def mean(self):
        return self.mean_bxn

    @property
    def logvar(self):
        return self.logvar_bxn

    @property
    def noise(self):
        return tf.random_normal(tf.shape(self.logvar))

    @property
    def sample(self):
        #return self.mean + tf.exp(0.5 * self.logvar) * self.noise
        return self.sample_bxn



    #def noise(self):
    #    return self.noise_bxn

    #def sample(self):
    #    return self.mean + tf.exp(0.5 * self.logvar) * self.noise()


def diag_gaussian_log_likelihood(z, mu=0.0, logvar=0.0):
    """Log-likelihood under a Gaussian distribution with diagonal covariance.
      Returns the log-likelihood for each dimension.  One should sum the
      results for the log-likelihood under the full multidimensional model.

    Args:
      z: The value to compute the log-likelihood.
      mu: The mean of the Gaussian
      logvar: The log variance of the Gaussian.

    Returns:
      The log-likelihood under the Gaussian model.
    """

    return -0.5 * (logvar + np.log(2 * np.pi) + \
                   tf.square((z - mu) / tf.exp(0.5 * logvar)))


def gaussian_pos_log_likelihood(unused_mean, logvar, noise):
  """Gaussian log-likelihood function for a posterior in VAE

  Note: This function is specialized for a posterior distribution, that has the
  form of z = mean + sigma * noise.

  Args:
    unused_mean: ignore
    logvar: The log variance of the distribution
    noise: The noise used in the sampling of the posterior.

  Returns:
    The log-likelihood under the Gaussian model.
  """
  # ln N(z; mean, sigma) = - ln(sigma) - 0.5 ln 2pi - noise^2 / 2
  return - 0.5 * (logvar + np.log(2 * np.pi) + tf.square(noise))


class DiagonalGaussianFromExisting(Gaussian):
    """Diagonal Gaussian with different constant mean and variances in each
    dimension.
    """

    def __init__(self, mean_bxn, logvar_bxn, var_min=0.0):
        self.mean_bxn = mean_bxn
        if var_min > 0.0:
            logvar_bxn = tf.log(tf.exp(logvar_bxn) + var_min)
            #logvar_bxn = tf.nn.relu(logvar_bxn) + tf.log(var_min)
        self.logvar_bxn = logvar_bxn

        self.noise_bxn = noise_bxn = tf.random_normal(tf.shape(logvar_bxn))
        #self.noise_bxn.set_shape([None, z_size])
        self.sample_bxn = mean_bxn + tf.exp(0.5 * logvar_bxn) * noise_bxn

    def logp(self, z=None):
        """Compute the log-likelihood under the distribution.

        Args:
          z (optional): value to compute likelihood for, if None, use sample.

        Returns:
          The likelihood of z under the model.
        """
        if z is None:
          z = self.sample

        # This is needed to make sure that the gradients are simple.
        # The value of the function shouldn't change.
        if z == self.sample_bxn:
          return gaussian_pos_log_likelihood(self.mean_bxn, self.logvar_bxn, self.noise_bxn)

        return diag_gaussian_log_likelihood(z, self.mean_bxn, self.logvar_bxn)

    # @property
    # def mean(self):
    #     return self.mean_bxn
    #
    # @property
    # def logvar(self):
    #     return self.logvar_bxn
    #
    # @property
    # def sample(self):
    #     return self.sample_bxn


class LearnableDiagonalGaussian(Gaussian):
    """Diagonal Gaussian with different constant mean and variances in each
    dimension.
    """

    def __init__(self, batch_size, z_size, name, var, trainable_mean=True, trainable_var=False):
        # MRK's fix, letting the mean of the prior to be trainable
        mean_init = 0.0
        num_steps = z_size[0]
        num_dim = z_size[1]
        z_mean_1xn = tf.get_variable(name=name+"/mean", shape=[1,1,num_dim],
                             initializer=tf.constant_initializer(mean_init), trainable=trainable_mean)
        self.mean_bxn = tf.tile(z_mean_1xn, tf.stack([batch_size, num_steps, 1] ))
        self.mean_bxn.set_shape([None] + z_size)

        # MRK, make Var trainable (for Controller prior)
        var_init = np.log(var)
        z_logvar_1xn = tf.get_variable(name=name+"/logvar", shape=[1,1,num_dim],
                                       initializer=tf.constant_initializer(var_init),
                                       trainable=trainable_var)
        self.logvar_bxn = tf.tile(z_logvar_1xn, tf.stack([batch_size, num_steps, 1]))
        self.logvar_bxn.set_shape([None] + z_size)
        # remove time axis if 1 (used for ICs)
        if num_steps == 1:
            self.mean_bxn = tf.squeeze(self.mean_bxn, axis=1)
            self.logvar_bxn = tf.squeeze(self.logvar_bxn, axis=1)

        self.noise_bxn = tf.random_normal(tf.shape(self.logvar_bxn))

    # @property
    # def mean(self):
    #     return self.mean_bxn
    #
    # @property
    # def logvar(self):
    #     return self.logvar_bxn
    #
    # @property
    # def sample(self):
    #     return self.sample_bxn
    # Not USED
    # def logp(self, z=None):
    #     """Compute the log-likelihood under the distribution.
    #
    #     Args:
    #       z (optional): value to compute likelihood for, if None, use sample.
    #
    #     Returns:
    #       The likelihood of z under the model.
    #     """
    #     if z is None:
    #       z = self.sample()
    #
    #     # This is needed to make sure that the gradients are simple.
    #     # The value of the function shouldn't change.
    #     if z == self.sample:
    #       return gaussian_pos_log_likelihood(self.mean, self.logvar, self.noise)
    #
    #     return diag_gaussian_log_likelihood(z, self.mean, self.logvar)


# Used for AR prior
class LearnableAutoRegressive1Prior(object):
  """AR(1) model where autocorrelation and process variance are learned
  parameters.  Assumed zero mean.

  """

  def __init__(self, batch_size, z_size,
               autocorrelation_taus, noise_variances,
               do_train_prior_ar_atau, do_train_prior_ar_nvar,
               name):
    """Create a learnable autoregressive (1) process.

    Args:
      batch_size: The size of the batch, i.e. 0th dim in 2D tensor of samples.
      z_size: The dimension of the distribution, i.e. 1st dim in 2D tensor.
      autocorrelation_taus: The auto correlation time constant of the AR(1)
      process.
        A value of 0 is uncorrelated gaussian noise.
      noise_variances: The variance of the additive noise, *not* the process
        variance.
      do_train_prior_ar_atau: Train or leave as constant, the autocorrelation?
      do_train_prior_ar_nvar: Train or leave as constant, the noise variance?
      num_steps: Number of steps to run the process.
      name: The name to prefix to learned TF variables.
    """

    # Note the use of the plural in all of these quantities.  This is intended
    # to mark that even though a sample z_t from the posterior is thought of a
    # single sample of a multidimensional gaussian, the prior is actually
    # thought of as U AR(1) processes, where U is the dimension of the inferred
    # input.
    size_bx1 = tf.stack([batch_size, 1])
    size__xu = [None, z_size]
    # process variance, the variance at time t over all instantiations of AR(1)
    # with these parameters.
    log_evar_inits_1xu = tf.expand_dims(tf.log(noise_variances), 0)
    self.logevars_1xu = logevars_1xu = \
        tf.Variable(log_evar_inits_1xu, name=name+"/logevars", dtype=tf.float32,
                    trainable=do_train_prior_ar_nvar)
    self.logevars_bxu = logevars_bxu = tf.tile(logevars_1xu, size_bx1)
    logevars_bxu.set_shape(size__xu) # tile loses shape

    # \tau, which is the autocorrelation time constant of the AR(1) process
    log_atau_inits_1xu = tf.expand_dims(tf.log(autocorrelation_taus), 0)
    self.logataus_1xu = logataus_1xu = \
        tf.Variable(log_atau_inits_1xu, name=name+"/logatau", dtype=tf.float32,
                    trainable=do_train_prior_ar_atau)

    # phi in x_t = \mu + phi x_tm1 + \eps
    # phi = exp(-1/tau)
    # phi = exp(-1/exp(logtau))
    # phi = exp(-exp(-logtau))
    phis_1xu = tf.exp(-tf.exp(-logataus_1xu))
    self.phis_bxu = phis_bxu = tf.tile(phis_1xu, size_bx1)
    phis_bxu.set_shape(size__xu)

    # process noise
    # pvar = evar / (1- phi^2)
    # logpvar = log ( exp(logevar) / (1 - phi^2) )
    # logpvar = logevar - log(1-phi^2)
    # logpvar = logevar - (log(1-phi) + log(1+phi))
    self.logpvars_1xu = \
        logevars_1xu - tf.log(1.0-phis_1xu) - tf.log(1.0+phis_1xu)
    self.logpvars_bxu = logpvars_bxu = tf.tile(self.logpvars_1xu, size_bx1)
    logpvars_bxu.set_shape(size__xu)

    # process mean (zero but included in for completeness)
    self.pmeans_bxu = pmeans_bxu = tf.zeros_like(phis_bxu)


  def logp_t(self, z_t_bxu, z_tm1_bxu=None):
    """Compute the log-likelihood under the distribution for a given time t,
    not the whole sequence.

    Args:
      z_t_bxu: sample to compute likelihood for at time t.
      z_tm1_bxu (optional): sample condition probability of z_t upon.

    Returns:
      The likelihood of p_t under the model at time t. i.e.
        p(z_t|z_tm1_bxu) = N(z_tm1_bxu * phis, eps^2)

    """
    if z_tm1_bxu is None:
      logp_tgtm1_bxu = diag_gaussian_log_likelihood(z_t_bxu, self.pmeans_bxu,
                                          self.logpvars_bxu)
    else:
      means_t_bxu = self.pmeans_bxu + self.phis_bxu * z_tm1_bxu
      logp_tgtm1_bxu = diag_gaussian_log_likelihood(z_t_bxu,
                                                    means_t_bxu,
                                                    self.logevars_bxu)
    return logp_tgtm1_bxu


def makeInitialState(state_dim, batch_size, name):
    #init_stddev = 1 / np.sqrt(float(state_dim))
    #init_initter = tf.random_normal_initializer(0.0, init_stddev, dtype=tf.float32)
    init_state = tf.get_variable(name + '_init_state', [1, state_dim],
                                 #initializer=init_initter,
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.float32, trainable=True)
    tile_dimensions = [batch_size, 1]
    init_state_tiled = tf.tile(init_state,
                               tile_dimensions,
                               name=name + '_init_state_tiled')
    return init_state_tiled

def tile_output( inputs, W, b, dropout=False, keep_prob=0.8, activation='relu' ):
    # old code, incorrect
    #tiled_W = tf.tile(W, [tf.shape(inputs)[0], 1])
    #tiled_W = tf.reshape(tiled_W, [-1, W.get_shape()[0], W.get_shape()[1]])
    
    #tiled_b = tf.tile(b, [tf.shape(inputs)[0], 1])
    #tiled_b = tf.reshape(tiled_b, [-1, b.get_shape()[0], b.get_shape()[1]])

    print( inputs )
    # moved dropout
    if dropout:
        print( 'Dropout applied' )
        inputs = tf.nn.dropout( inputs, keep_prob )

    # from tpu branch
    tiled_W = tf.tile(tf.expand_dims(W, 0), [tf.shape(inputs)[0], 1, 1])
    tiled_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(inputs)[0], 1, 1])

    if activation=='relu':
        output = tf.nn.relu( tf.matmul(inputs, tiled_W) + tiled_b )
    elif activation=='softplus':
        output = tf.nn.softplus( tf.matmul(inputs, tiled_W) + tiled_b )
    elif activation=='sigmoid':
        output = tf.sigmoid( tf.matmul(inputs, tiled_W) + tiled_b )
    elif activation=='scaled_sigmoid':
        output = tf.matmul(inputs, tiled_W) + tiled_b
        output = tf.multiply(tf.sigmoid(output), 20.0)
        output = tf.clip_by_value(output, clip_value_min=1e-5, clip_value_max = 20.0-1e-5)
    else:
        output = tf.matmul(inputs, tiled_W) + tiled_b

    return output

def split_linear_transform_zigamma( inputs, output_size, W, b ):
    # get size for weights related to gamma distribution and related to q for ZIG        
    gamma_size = tf.cast(output_size*(2/3), tf.int64)
    q_size = tf.cast(output_size*(1/3), tf.int64)
    gamma_W, q_W = tf.split( W, [gamma_size, q_size], axis=1 )
    gamma_b, q_b = tf.split( b, [gamma_size, q_size], axis=1 )
    tiled_gamma_W = tf.tile(tf.expand_dims(gamma_W, 0), [tf.shape(inputs)[0], 1, 1])
    tiled_q_W = tf.tile(tf.expand_dims(q_W, 0), [tf.shape(inputs)[0], 1, 1])
    tiled_gamma_b = tf.tile(tf.expand_dims(gamma_b, 0), [tf.shape(inputs)[0], 1, 1])
    tiled_q_b = tf.tile(tf.expand_dims(q_b, 0), [tf.shape(inputs)[0], 1, 1])
    output_gamma = tf.matmul(inputs, tiled_gamma_W) + tiled_gamma_b
    output_q = tf.matmul(inputs, tiled_q_W) + tiled_q_b
    return output_gamma, output_q, gamma_size, q_size

def trainable_scaled_sigmoid_transform( output, scaled_sigmoid_collection, output_size, scaled_sigmoid_init=20.0, trainable=True ):
    """
    output: 3d tf tensor, which is usually the linear readout from factors
    scaled_sigmoid_collection: string. Name of the collection that the sigmoid scales will be assigned to
    gamma_size: scalar. Number of output units (which is 2x number of neurons if using gamma)
    scaled_sigmoid_init: scalar. Initial value of sigmoid scales (same for all output units)
    trainable: Boolean. Whether to make sigmoid scales trainable
    """
    # ================================
    # set initializers for gamma scale and get appropriate collection for it
    # ================================
    print('Doing scaled sigmoid transform in trainable_scaled_sigmoid_transform function')
    w_collections = [tf.GraphKeys.GLOBAL_VARIABLES, scaled_sigmoid_collection]
    # initialize sigmoid scale for each individual output units
    sigmoid_scale_init = tf.multiply(tf.ones(shape=[1,output_size]), scaled_sigmoid_init)
    sigmoid_scale_name = 'sigmoid_scale'
    # get variable for sigmoid scale for all output units based on the init values we set
    sigmoid_scale = tf.get_variable( sigmoid_scale_name, initializer=sigmoid_scale_init, trainable=trainable, dtype=tf.float32, collections=w_collections )
    output_nl = tf.sigmoid(output)
    # scale the sigmoid output with sigmoid scale
    output_nl = output_nl*sigmoid_scale
    # clip the output of sigmoid transform to avoid the outputs getting literally zero due to numerical precision
    # clip value max is not important. Can remove but it doesn't have much effect, at all.
    output_nl = tf.clip_by_value(output_nl, clip_value_min=1e-5, clip_value_max = sigmoid_scale-1e-5)
    return output_nl

class FeedforwardTimeVarying(object):
    
    def __init__(self, inputs, output_size, output_name, collections, transform_name=None, do_bias=True,normalized=False, keep_prob=1.0, zigamma=False):
        num_timesteps = tf.shape( inputs )[1]
        input_size = inputs.get_shape().as_list()[2]
        outputs = []
        self.keep_prob = keep_prob
        # =======================
        # get weights collections
        # =======================
        
        w_collections = [tf.GraphKeys.GLOBAL_VARIABLES, "norm-variables"]
        
        # ===================================
        # set initializers for weights/biases
        # ===================================
        
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.zeros_initializer()
        
        # ==============================================
        # create weights/biases variables for all layers
        # ==============================================
        
        # add weights to collections
        w_collections += collections
        
        
        # define names for all weights/biases
        w_h1_name = 'fac_2_rates_h1/W'
        b_h1_name = 'fac_2_rates_h1/b'
        w_h2_name = 'fac_2_rates_h2/W'
        b_h2_name = 'fac_2_rates_h2/b'
        w_oL_name = 'fac_2_rates_oL/W'
        b_oL_name = 'fac_2_rates_oL/b'
        
        print( 'I am a happy feedforward network that will find a nice nonlinear mapping for you :)' )

        # ==============================================
        # create weights/biases variables for all layers
        # ==============================================

        # create hidden layer I weights/biases
        w_h1 = tf.get_variable( w_h1_name, [ input_size, input_size ], initializer=w_init, dtype=tf.float32, collections=w_collections )
        b_h1 = tf.get_variable( b_h1_name, [ 1, input_size ], initializer=b_init, dtype=tf.float32 )
        # create hidden layer II weights/biases
        w_h2 = tf.get_variable( w_h2_name, [ input_size, input_size ], initializer=w_init, dtype=tf.float32, collections=w_collections )
        b_h2 = tf.get_variable( b_h2_name, [ 1, input_size ], initializer=b_init, dtype=tf.float32 )
        # create output layer weights/biases
        w_oL = tf.get_variable( w_oL_name, [ input_size, output_size ], initializer=w_init, dtype=tf.float32, collections=w_collections )
        b_oL = tf.get_variable( b_oL_name, [ 1, output_size ], initializer=b_init, dtype=tf.float32 )

        # =====================================
        # compute output of feedforward network
        # =====================================
        
        output_hL1 = tile_output(inputs, w_h1, b_h1, activation='relu', dropout=True, keep_prob=self.keep_prob)
        output_hL2 = tile_output(output_hL1, w_h2, b_h2, activation='relu', dropout=True, keep_prob=self.keep_prob)
        # deal with ziGamma split
        if zigamma is True:
            # get size for weights related to gamma distribution and related to q for ZIG        
            gamma_W_size = tf.cast(output_size*(2/3), tf.int64)
            q_size = tf.cast(output_size*(1/3), tf.int64)
            gamma_W, q_W = tf.split( w_oL, [gamma_W_size, q_size], axis=1 )
            gamma_b, q_b = tf.split( b_oL, [gamma_W_size, q_size], axis=1 )
            #output_gamma = tile_output(output_hL2, gamma_W, gamma_b, activation='softplus', dropout=False)
            output_gamma = tile_output(output_hL2, gamma_W, gamma_b, activation='scaled_sigmoid', dropout=False)
            output_q = tile_output(output_hL2, q_W, q_b, activation='sigmoid', dropout=False)
            # this is temporary. Do NOT un-comment
            #output_q = tf.where(tf.math.less(output_q, 0.1), tf.math.multiply(tf.ones_like(output_q), 0.01), tf.math.multiply(tf.ones_like(output_q), 0.3))
            output_q = tf.clip_by_value(output_q, clip_value_min=1e-5, clip_value_max=1-1e-5)
            output = tf.concat([output_gamma, output_q], 2)
            self.output = output
            print('Using FFN for ZiGamma')
        else:
            output = tile_output(output_hL2, w_oL, b_oL, activation='softplus', dropout=False)
            self.output = output



class LinearTimeVarying(object):
        # self.output = linear transform
    # self.output_nl = nonlinear transform

    def __init__(self, inputs, output_size, transform_name, nonlinearity=None,
                 collections=None, W=None, b=None, normalized=False, do_bias=True,
                 scaled_sigmoid_collection=None, zigamma=False):
        num_timesteps = tf.shape(inputs)[1]
        # must return "as_list" to get ints
        input_size = inputs.get_shape().as_list()[2]
        outputs = []
        outputs_nl = []
        # use any matrices provided, if they exist
        if W is not None and b is None:
            raise ValueError('LinearTimeVarying: must provide either W and b, or neither')
        if W is None and b is not None:
            raise ValueError('LinearTimeVarying: must provide either W and b, or neither')

        if W is None and b is None:
            W, b = init_linear_transform(input_size, output_size, name=transform_name,
                                         collections=collections, normalized=normalized,
                                         do_bias=do_bias)
        self.W = W
        self.b = b

        if nonlinearity is 'scaled_sigmoid' and scaled_sigmoid_collection is None:
            raise ValueError('LinearTimeVarying: must provide scaled_sigmoid_collection when using scaled_sigmoid as nonlinearity')

        # inputs_permuted = tf.transpose(inputs, perm=[1, 0, 2])
        # initial_outputs = tf.TensorArray(dtype=tf.float32, size=num_timesteps, name='init_linear_outputs')
        # initial_outputs_nl = tf.TensorArray(dtype=tf.float32, size=num_timesteps, name='init_nl_outputs')

        # MRK: replaced tf.while_loop with a simple tf.matmul
        #tiled_W = tf.tile(W, [tf.shape(inputs)[0], 1])
        #tiled_W = tf.reshape(tiled_W, [-1, W.get_shape()[0], W.get_shape()[1]])
        #tiled_b = tf.tile(b, [tf.shape(inputs)[0], 1])
        #tiled_b = tf.reshape(tiled_b, [-1, b.get_shape()[0], b.get_shape()[1]])
        #output = tf.matmul(inputs, tiled_W) + tiled_b

        if zigamma is True:
            output_gamma, output_q, gamma_size, q_size = split_linear_transform_zigamma( inputs, output_size, self.W, self.b )
            # apply sigmoid to output_q in all cases
            output_q_nl = tf.sigmoid(output_q)
            output_q_nl = tf.clip_by_value(output_q_nl, clip_value_min=1e-5, clip_value_max=1-1e-5)
            if nonlinearity is 'exp':
                output_gamma_nl = tf.exp(output_gamma)
                self.output_nl = tf.concat([output_gamma_nl, output_q_nl], 2)
            elif nonlinearity is 'scaled_sigmoid':
                output_gamma_nl = trainable_scaled_sigmoid_transform(output_gamma, scaled_sigmoid_collection, gamma_size, scaled_sigmoid_init=20.0, trainable=True)
                self.output_nl = tf.concat([output_gamma_nl, output_q_nl], 2)
            self.output = tf.concat([output_gamma, output_q_nl], 2)
        else:            
            tiled_W = tf.tile(tf.expand_dims(W, 0), [tf.shape(inputs)[0], 1, 1])
            tiled_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(inputs)[0], 1, 1])
            output = tf.matmul(inputs, tiled_W) + tiled_b

            if nonlinearity is 'exp':
                output_nl = tf.exp(output)
                self.output_nl = output_nl
            elif nonlinearity is 'scaled_sigmoid':
                scaled_sigmoid_init=20.0 # can make this a HP
                output_nl = trainable_scaled_sigmoid_transform(output, scaled_sigmoid_collection, output_size, scaled_sigmoid_init=scaled_sigmoid_init, trainable=True) 
                self.output_nl = output_nl
            self.output = output

class LinearTimeVarying_OLD(object):
    # self.output = linear transform
    # self.output_nl = nonlinear transform

    def __init__(self, inputs, output_size, transform_name, nonlinearity=None,
                 collections=None, W=None, b=None, normalized=False, do_bias=True):
        # expand for 1 time step transform
        #if len(inputs.get_shape()) == 2:
        #    inputs = tf.expand_dims(inputs, [1])
        num_timesteps = tf.shape(inputs)[1]
        # must return "as_list" to get ints
        input_size = inputs.get_shape().as_list()[2]
        outputs = []
        outputs_nl = []
        # use any matrices provided, if they exist
        if W is not None and b is None:
            raise ValueError('LinearTimeVarying: must provide either W and b, or neither')
        if W is None and b is not None:
            raise ValueError('LinearTimeVarying: must provide either W and b, or neither')

        if W is None and b is None:
            W, b = init_linear_transform(input_size, output_size, name=transform_name,
                                         collections=collections, normalized=normalized,
                                         do_bias=do_bias)
        self.W = W
        self.b = b

        inputs_permuted = tf.transpose(inputs, perm=[1, 0, 2])
        initial_outputs = tf.TensorArray(dtype=tf.float32, size=num_timesteps, name='init_linear_outputs')
        initial_outputs_nl = tf.TensorArray(dtype=tf.float32, size=num_timesteps, name='init_nl_outputs')

        # keep going until the number of timesteps
        def condition(t, *args):
            return t < num_timesteps

        def iteration(t_, output_, output_nl_):
            # apply linear transform to input at this timestep
            ## cur = tf.gather(inputs, t, axis=1)
            # axis is not supported in 'gather' until 1.3
            # cur = tf.gather(inputs_permuted, t)
            cur = inputs_permuted[t_, :, :]
            output_this_step = tf.matmul(cur, self.W) + self.b
            output_ = output_.write(t_, output_this_step)
            if nonlinearity is 'exp':
                output_nl_ = output_nl_.write(t_, tf.exp(output_this_step))
            return t_ + 1, output_, output_nl_

        i = tf.constant(0)
        t, output, output_nl = tf.while_loop(condition, iteration, \
                                             [i, initial_outputs, initial_outputs_nl])

        self.output = tf.transpose(output.stack(), perm=[1, 0, 2])
        self.output_nl = tf.transpose(output_nl.stack(), perm=[1, 0, 2])

        ## this is old code for the linear time varying transform
        # was replaced by the above tf.while_loop

        # for step_index in range(tensor_shape[1]):
        #    gred = inputs[:, step_index, :]
        #    fout = tf.matmul(gred, self.W) + self.b
        #    # add a leading dimension for concatenating later
        #    outputs.append( tf.expand_dims( fout , 1) )
        #    if nonlinearity is 'exp':
        #        nlout = tf.exp(fout)
        #        # add a leading dimension for concatenating later
        #        outputs_nl.append( tf.expand_dims( nlout , 1) )
        # concatenate the created list into the factors
        # self.output = tf.concat(outputs, axis=1, name=output_name)
        # if nonlinearity is 'exp':
        #    self.output_nl = tf.concat(outputs_nl, axis=1, name=output_name)


class KLCost_GaussianGaussian(object):
    """log p(x|z) + KL(q||p) terms for Gaussian posterior and Gaussian prior. See
    eqn 10 and Appendix B in VAE for latter term,
    http://arxiv.org/abs/1312.6114

    The log p(x|z) term is the reconstruction error under the model.
    The KL term represents the penalty for passing information from the encoder
    to the decoder.
    To sample KL(q||p), we simply sample
          ln q - ln p
    by drawing samples from q and averaging.
    """

    def __init__(self, z, prior_z):
        """Create a lower bound in three parts, normalized reconstruction
        cost, normalized KL divergence cost, and their sum.

        E_q[ln p(z_i | z_{i+1}) / q(z_i | x)
           \int q(z) ln p(z) dz = - 0.5 ln(2pi) - 0.5 \sum (ln(sigma_p^2) + \
              sigma_q^2 / sigma_p^2 + (mean_p - mean_q)^2 / sigma_p^2)

           \int q(z) ln q(z) dz = - 0.5 ln(2pi) - 0.5 \sum (ln(sigma_q^2) + 1)

        Args:
          zs: posterior z ~ q(z|x)
          prior_zs: prior zs
        """
        # L = -KL + log p(x|z), to maximize bound on likelihood
        # -L = KL - log p(x|z), to minimize bound on NLL
        # so 'KL cost' is postive KL divergence
        kl_b = 0.0
        # for z, prior_z in zip(zs, prior_zs):
        assert isinstance(z, Gaussian)
        assert isinstance(prior_z, Gaussian)
        # ln(2pi) terms cancel
        kl_b += 0.5 * tf.reduce_sum(
            prior_z.logvar - z.logvar
            + tf.exp(z.logvar - prior_z.logvar)
            + tf.square((z.mean - prior_z.mean) / tf.exp(0.5 * prior_z.logvar))
            - 1.0, [1])

        self.kl_cost_b = tf.reduce_sum(kl_b, [1]) if len(kl_b.get_shape()) == 2 else kl_b
        self.kl_cost = tf.reduce_mean(kl_b)

        #self.kl_cost = tf.reduce_mean(kl_b)

# Used for AR prior
class KLCost_GaussianGaussianProcessSampled(object):
  """ log p(x|z) + KL(q||p) terms for Gaussian posterior and Gaussian process
  prior via sampling.

  The log p(x|z) term is the reconstruction error under the model.
  The KL term represents the penalty for passing information from the encoder
  to the decoder.
  To sample KL(q||p), we simply sample
        ln q - ln p
  by drawing samples from q and averaging.
  """

  def __init__(self, post_zs, prior_z_process):
    """Create a lower bound in three parts, normalized reconstruction
    cost, normalized KL divergence cost, and their sum.

    Args:
      post_zs: posterior z ~ q(z|x)
      prior_z_process: prior AR(1) process
    """
    #assert len(post_zs) > 1, "GP is for time, need more than 1 time step."
    #assert isinstance(prior_z_process, GaussianProcess), "Must use GP."

    # L = -KL + log p(x|z), to maximize bound on likelihood
    # -L = KL - log p(x|z), to minimize bound on NLL
    # so 'KL cost' is postive KL divergence

    # sample from the posterior for all time points and dimensions
    post_zs_sampled = post_zs.sample
    # sum KL over time and dimension axis
    logq_bxu = tf.reduce_sum(post_zs.logp(post_zs_sampled), [1,2])

    logp_bxu = 0
    num_steps = post_zs.mean.get_shape()[1]
    for i in range(num_steps):
      # posterior is independent in time, prior is not
      if i == 0:
          z_tm1_bxu = None
      else:
          z_tm1_bxu = post_zs_sampled[:, i-1, :]
      logp_bxu += tf.reduce_sum(prior_z_process.logp_t(
          post_zs_sampled[:,i,:], z_tm1_bxu), [1])

    kl_b = logq_bxu - logp_bxu
    self.kl_cost_b = kl_b


"""Wrappers for primitive Neural Net (NN) Operations."""

import numbers
#from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops



# dropout that also returns the binary mask
def dropout(x, keep_prob, noise_shape=None, seed=None, name=None,
            binary_tensor=None):  # pylint: disable=invalid-name
    """Computes dropout.
    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.
    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.
    Args:
      x: A floating point tensor.
      keep_prob: A scalar `Tensor` with the same type as x. The probability
        that each element is kept.
      noise_shape: A 1-D `Tensor` of type `int32`, representing the
        shape for randomly generated keep/drop flags.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      name: A name for this operation (optional).
    Returns:
      A Tensor of the same shape of `x`.
    Raises:
      ValueError: If `keep_prob` is not in `(0, 1]` or if `x` is not a floating
        point tensor.
    """
    with ops.name_scope(name, "dropout", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)

        # Only apply random dropout if mask is not provided

        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)

        keep_prob = keep_prob if binary_tensor is None else 1 - keep_prob

        keep_prob = ops.convert_to_tensor(keep_prob,
                                          dtype=x.dtype,
                                          name="keep_prob")

        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if binary_tensor is None:
            # Do nothing if we know keep_prob == 1
            if tensor_util.constant_value(keep_prob) == 1:
                return x, None

            noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
            # uniform [keep_prob, 1.0 + keep_prob)
            random_tensor = keep_prob
            random_tensor += random_ops.random_uniform(noise_shape,
                                                       seed=seed,
                                                       dtype=x.dtype)
            # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            binary_tensor = math_ops.floor(random_tensor)
        else:
            # check if binary_tensor is a tensor with right shape
            binary_tensor = math_ops.cast(binary_tensor, dtype=x.dtype)
            # pass

        ret = math_ops.div(x, keep_prob) * binary_tensor
        # if context.in_graph_mode():
        ret.set_shape(x.get_shape())
        return ret, binary_tensor
    
def shuffle_spikes_in_time(data_bxtxd, w):
    """Shuffle the spikes in the temporal dimension.  This is useful to
    help the LFADS system avoid overfitting to individual spikes or fast
    oscillations found in the data that are irrelevant to behavior. A
    pure 'tabula rasa' approach would avoid this, but LFADS is sensitive
    enough to pick up dynamics that you may not want.

    Args:
      data_bxtxd: numpy array of spike count data to be shuffled.
    Returns:
    S_bxtxd, a numpy array with the same dimensions and contents as
      data_bxtxd, but shuffled appropriately.

    """

    B, T, N = data_bxtxd.shape
    #w = self.hps.temporal_spike_jitter_width

    if w == 0:
        return data_bxtxd

    max_counts = int( np.max(data_bxtxd) )
    S_bxtxd = np.zeros([B,T,N])

    # Intuitively, shuffle spike occurances, 0 or 1, but since we have counts,
    # Do it over and over again up to the max count.
    for mc in range(1,max_counts+1):
        idxs = np.nonzero(data_bxtxd >= mc)

        data_ones = np.zeros_like(data_bxtxd)
        data_ones[data_bxtxd >= mc] = 1

        nfound = len(idxs[0])
        shuffles_incrs_in_time = np.random.randint(-w, w, size=nfound)

        shuffle_tidxs = idxs[1].copy()
        shuffle_tidxs += shuffles_incrs_in_time

        # Reflect on the boundaries to not lose mass.
        shuffle_tidxs[shuffle_tidxs < 0] = -shuffle_tidxs[shuffle_tidxs < 0]
        shuffle_tidxs[shuffle_tidxs > T-1] = \
                                             (T-1)-(shuffle_tidxs[shuffle_tidxs > T-1] -(T-1))

        for iii in zip(idxs[0], shuffle_tidxs, idxs[2]):
            S_bxtxd[iii] += 1

    return S_bxtxd


class zeroInflatedGamma(object):
        # self.output = linear transform
    # self.output_nl = nonlinear transform

    def __init__(self, alpha, beta, q, s_min):
        num_timesteps = tf.shape(alpha)[1]
        # must return "as_list" to get ints
        #input_size = alpha.get_shape().as_list()[2]
        self.alpha = alpha
        self.beta = beta
        self.q = q
        #self.s_min = 0.1 # or 0.0307-5e-5 for inequal or (0.1/3)-1e-5 for equal interval sparse sampl
        self.s_min = s_min

    def log_prob_ZIG(self, x):
        # give where spike = 0 a value of 1, to be able to use log_prob gamma.
        # give where spike is not 0 a value of x-s_min
        adjust_x = tf.where(tf.math.equal(x, 0.0), tf.ones_like(x), x-self.s_min)
        # compute log-likelihood element-wise. Where spike=0 now has a inaccurate value
        loglikelihood_adj_gamma = tfd.Gamma( self.alpha, self.beta ).log_prob( adjust_x )
        # now convert those inaccurate value to be log(1-q)
        # add log(q) to places where x>s_min
        self.loglikelihood = tf.where(tf.math.equal(x, 0.0), tf.math.log(1-self.q), loglikelihood_adj_gamma+tf.math.log(self.q))
        return self.loglikelihood
    
