from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from customcells import GRUCell


class BidirectionalDynamicRNN(object):
    def __init__(self, state_dim, batch_size, name, sequence_lengths,
                 inputs=None, initial_state=None, rnn_type='gru',
                 clip_value = None, recurrent_collections = None):

        if initial_state is None:
            # need initial states for fw and bw
            self.init_initter = tf.zeros_initializer()
            self.init_h_fw = tf.get_variable(name + '_init_h_fw', [1, state_dim],
                                             initializer=self.init_initter,
                                             dtype=tf.float32)
            self.init_h_bw = tf.get_variable(name + '_init_h_bw', [1, state_dim],
                                             initializer=self.init_initter,
                                             dtype=tf.float32)
            # lstm has a second parameter c
            if rnn_type.lower() == 'lstm':
                self.init_c_fw = tf.get_variable(name + '_init_c_fw', [1, state_dim],
                                                 initializer=self.init_initter,
                                                 dtype=tf.float32)
                self.init_c_bw = tf.get_variable(name + '_init_c_bw', [1, state_dim],
                                                 initializer=self.init_initter,
                                                 dtype=tf.float32)

            tile_dimensions = [batch_size, 1]

            # tile the h param
            self.init_h_fw_tiled = tf.tile(self.init_h_fw,
                                           tile_dimensions, name=name + '_h_fw_tile')
            self.init_h_bw_tiled = tf.tile(self.init_h_bw,
                                           tile_dimensions, name=name + '_h_bw_tile')
            # tile the c param if needed
            if rnn_type.lower() == 'lstm':
                self.init_c_fw_tiled = tf.tile(self.init_c_fw,
                                               tile_dimensions, name=name + '_c_fw_tile')
                self.init_c_bw_tiled = tf.tile(self.init_c_bw,
                                               tile_dimensions, name=name + '_c_bw_tile')

            # do tupling if needed
            if rnn_type.lower() == 'lstm':
                # lstm state is a tuple
                init_fw = tf.contrib.rnn.LSTMStateTuple(self.init_c_fw_tiled, self.init_h_fw_tiled)
                init_bw = tf.contrib.rnn.LSTMStateTuple(self.init_c_bw_tiled, self.init_h_bw_tiled)
                self.init_fw = init_fw
                self.init_bw = init_bw
            else:
                self.init_fw = self.init_h_fw_tiled
                self.init_bw = self.init_h_bw_tiled
                
        else:  # if initial state is None
            self.init_fw, self.init_bw = initial_state

        # pick your cell
        if rnn_type.lower() == 'lstm':
            self.cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=state_dim,
                                                state_is_tuple=True)
            self.cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=state_dim,
                                                state_is_tuple=True)
        elif rnn_type.lower() == 'gru':
            self.cell_fw = tf.nn.rnn_cell.GRUCell(num_units=state_dim)
            self.cell_bw = tf.nn.rnn_cell.GRUCell(num_units=state_dim)

        elif rnn_type.lower() == 'customgru':
            self.cell_fw = GRUCell(num_units = state_dim,
                                  clip_value = clip_value,
                                  recurrent_collections = recurrent_collections
                                  )
            self.cell_bw = GRUCell(num_units = state_dim,
                                  clip_value = clip_value,
                                  recurrent_collections = recurrent_collections
                                  )
        else:
            raise ValueError("Didn't understand rnn_type '%s'."%(rnn_type))

        if inputs is None:
            inputs = tf.zeros([batch_size, sequence_lengths, 1],
                              dtype=tf.float32)
        self.states, self.last = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell_fw,
            cell_bw=self.cell_bw,
            dtype=tf.float32,
            inputs=inputs,
            initial_state_fw=self.init_fw,
            initial_state_bw=self.init_bw,
        )

        # concatenate the outputs of the encoders (h only) into one vector
        self.last_fw, self.last_bw = self.last

        if rnn_type.lower() == 'lstm':
            self.last_fw.h, _ = self.last_fw
            self.last_bw.h, _ = self.last_bw
            self.last_tot = tf.concat(axis=1, values=[self.last_fw.h, self.last_bw.h])
        else:
            self.last_tot = tf.concat(axis=1, values=[self.last_fw, self.last_bw])

''' # Not used:

class DynamicRNN(object):
    def __init__(self, state_dim, batch_size, name, sequence_lengths,
                 inputs=None, initial_state=None, rnn_type='gru',
                 clip_value = None, recurrent_collections = None):
#                 output_keep_prob=1.0,
#                 input_keep_prob=1.0):
        if initial_state is None:
            # need initial states for fw and bw
            self.init_stddev = 1 / np.sqrt(float(state_dim))
            self.init_initter = tf.random_normal_initializer(0.0, self.init_stddev, dtype=tf.float32)

            self.init_h = tf.get_variable(name + '_init_h', [1, state_dim],
                                          initializer=self.init_initter,
                                          dtype=tf.float32)
            if rnn_type.lower() == 'lstm':
                self.init_c = tf.get_variable(name + '_init_c', [1, state_dim],
                                              initializer=self.init_initter,
                                              dtype=tf.float32)

            tile_dimensions = [batch_size, 1]

            self.init_h_tiled = tf.tile(self.init_h,
                                        tile_dimensions, name=name + '_tile')

            if rnn_type.lower() == 'lstm':
                self.init_c_tiled = tf.tile(self.init_c,
                                            tile_dimensions, name=name + '_tile')

            if rnn_type.lower() == 'lstm':
                # tuple for lstm
                self.init = tf.contrib.rnn.LSTMStateTuple(self.init_c_tiled, self.init_h_tiled)
            else:
                #self.init = self.init_h_tiled
                self.init = tf.zeros_like( self.init_h_tiled )
        else:  # if initial state is None
            self.init = initial_state


        # pick your cell
        if rnn_type.lower() == 'lstm':
            self.cell = tf.nn.rnn_cell.LSTMCell(num_units=state_dim,
                                                state_is_tuple=True)
        elif rnn_type.lower() == 'gru':
            self.cell = tf.nn.rnn_cell.GRUCell(num_units=state_dim)
            #self.cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=state_dim)
        elif rnn_type.lower() == 'customgru':
            self.cell = GRUCell(num_units = state_dim,
                                      #batch_size = batch_size,
                                      clip_value = clip_value,
                                      recurrent_collections = recurrent_collections
                                      )
        else:
            raise ValueError("Didn't understand rnn_type '%s'."%(rnn_type))


        # add dropout if requested
        #self.cell = tf.contrib.rnn.DropoutWrapper(
        #        self.cell, output_keep_prob=output_keep_prob)

        # for some reason I can't get dynamic_rnn to work without inputs
        #  so generate fake inputs if needed...
        if inputs is None:
            inputs = tf.zeros([batch_size, sequence_lengths, 1],
                              dtype=tf.float32)
        # call dynamic_rnn
        #inputs.set_shape((None, sequence_lengths, inputs.get_shape()[2]))
        self.states, self.last = tf.nn.dynamic_rnn(
            cell=self.cell,
            dtype=tf.float32,
            #sequence_length = sequence_lengths,
            inputs=inputs,
            initial_state=self.init,
        )
'''

