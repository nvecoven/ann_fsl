import tensorflow as tf
import numpy as np
import datetime
import json
import pickle
import os
import time

class CustomModel():
    def __init__(self, save_path, load = False, name = None, params = None, not_sequential_data = None, **kwargs):
        self.variables_d = {}
        self.variables = []
        self.queue_load_weights = {}
        self.whole_variables_d = {}

        self.not_sequentiel_data = not_sequential_data
        if not_sequential_data is None:
            self.not_sequential_data = ['_lengths', '_max_lengths']
        else:
            self.not_sequential_data += ['_lengths', '_max_lengths']

        self.kwargs = kwargs
        layers = self.define_layers(params = params)
        self.create_optimizer(params)
        layers['opt'] = self.opt

        self.checkpoint = tf.train.Checkpoint(**layers)

        self.to_pickle = {}
        if not load:
            self.to_pickle['name'] = name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.to_pickle['name'] = name

        if not load:
            self.to_pickle['batch_nbr'] = 1
            self.to_pickle['save_path'] = save_path
        else:
            self.restore_model(save_path)

        self.create_writers()

        self.losses_d = {}
        self.outputs_d = {}

    def define_layers(self, params):
        raise NotImplementedError("Please Implement this method")

    def create_writers(self):
        self.train_sum = tf.summary.create_file_writer(self.to_pickle['save_path'] + self.to_pickle['name'] +
                                                       "/log/train")
        self.test_sum = tf.summary.create_file_writer(self.to_pickle['save_path'] + self.to_pickle['name'] +
                                                      "/log/test")

    def create_optimizer(self, params):
        self.opt = tf.optimizers.Adam(learning_rate = 1e-4)

    def store_model(self):
        dir = self.to_pickle['save_path'] + self.to_pickle['name']
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.checkpoint.save(dir+"/params/")
        json.dump(self.to_pickle, open(dir + "/infos", "w"))

    def restore_model(self, path):
        self.checkpoint.restore(tf.train.latest_checkpoint(path + self.to_pickle['name'] + "/params/"))
        self.to_pickle = json.load(open(path + self.to_pickle['name'] + "/infos", "r"))

    def standardize(self, d):
        d = d.copy()
        try:
            assert (all(len(d[el]) == len(d[list(d.keys())[0]]) for el in d))
        except AssertionError:
            print("Each list included in the dictionnary must have the same number of trajectories (i.e."
                  "number of samples in the dataset)")
            raise
        check_lengths = np.array([len(el) for el in d[list(d.keys())[0]]])
        l = d[list(d.keys())[0]]
        max_size = np.max(check_lengths)
        to_return_lengths = np.zeros((len(l), max_size, 1), dtype=np.float32)
        for cnt, leng in enumerate(check_lengths):
            to_return_lengths[cnt, :leng, :] = 1

        for key in d:
            l = d[key]
            lengths = np.array([len(el) for el in l])
            try:
                assert(all(lengths == check_lengths))
            except AssertionError:
                print("Each corresponding trajectory of each list included in the dictionnary must be of the same size (i.e."
                      "number of time-steps for a given trajectory)")
                raise
            max_size = np.max(lengths)

            if not np.isreal(l[0][0][0]):
                to_return = np.zeros((len(l), max_size) + l[0].shape[1:], dtype=object)
            else:
                to_return = np.zeros((len(l), max_size) + l[0].shape[1:], dtype = np.float32)
            for cnt, el in enumerate(l):
                to_return[cnt][:lengths[cnt]] = el

            #Put default image as the first one of the first trajectory
            if not np.any(np.isreal(l[0][0])):
                to_return[np.where(to_return == 0)] = l[0][0][0]

            d[key] = to_return
        d['_lengths'] = np.reshape(check_lengths.astype(np.float32), (-1, 1))
        d['_lengths_mult'] = to_return_lengths
        d['_max_lengths'] = np.ones_like(d['_lengths'], dtype = np.int32) * max_size
        return d

    def destandardize(self, d):
        to_return = {}
        for key in d:
            if not (key == "_lengths" or key == "_lengths_mult"):
                to_return[key] = [el[:int(el2[0])] for el, el2 in zip(d[key], d['_lengths'])]
        return to_return

    def get_default_state(self, samples):
        return None

    @tf.function
    def forward_n(self, samples, prev_state, func, n, **kwargs):
        nbr_steps = n
        # first = True
        state = prev_state
        if prev_state is None:
            state = self.get_default_state(samples)

        sample_0 = {el: samples[el][:, 0, :] if not el in self.not_sequential_data else samples[el] for el in samples}
        output, state = func(sample_0, state, **kwargs)
        keys = output.keys()
        shapes = [tf.shape(output[el]) for el in keys]
        shapes_mult = [tf.reduce_prod(el) for el in shapes]

        outputs = tf.TensorArray(tf.float32, nbr_steps)

        flatten_output = tf.concat([tf.reshape(output[el], [-1, 1]) for el in output], axis = 1)
        outputs = outputs.write(0, flatten_output)

        for j in tf.range(1,nbr_steps):
            sample_j = {el: samples[el][:, j, :] if not el in self.not_sequential_data else samples[el] for el in samples}
            output, state = func(sample_j, state)
            flatten_output = tf.concat([tf.reshape(output[el], [-1, 1]) for el in output], axis = 1)
            outputs = outputs.write(j, flatten_output)

        outputs = tf.transpose(outputs.stack(), [1, 0, 2])
        to_return = {}
        cursor = 0
        for cnt,el in enumerate(keys):
            end_cursor = cursor+shapes_mult[cnt]//shapes[cnt][0]
            to_return[el] = tf.reshape(outputs[:,:,cursor:end_cursor], tf.concat([[shapes[cnt][0], nbr_steps], shapes[cnt][1:]], axis =0))
            cursor = end_cursor
        return to_return, state

    @tf.function
    def complete_unroll_forward_n(self, samples, prev_state, func, n, **kwargs):
        nbr_steps = n
        first = True
        state = prev_state
        if prev_state is None:
            state = self.get_default_state(samples)
        for j in range(nbr_steps):
            sample_j = {el: samples[el][:, j, :] if not el in self.not_sequential_data else samples[el] for el in
                        samples}
            output, state = func(sample_j, state, **kwargs)
            if first:
                first = False
                outputs = {el: tf.TensorArray(tf.float32, nbr_steps) for el in output}
            for el in outputs:
                outputs[el] = outputs[el].write(j, output[el])
        for el in outputs:
            outputs[el] = outputs[el].stack()
            outputs[el] = tf.transpose(outputs[el], tf.concat([[1, 0], tf.range(tf.size(outputs[el].shape[2:]))+2], axis = 0))

        return outputs, state

    def dynamic_forward_n(self, samples, prev_state, func, n, **kwargs):
        nbr_steps = n
        first = True
        state = prev_state
        if prev_state is None:
            state = self.get_default_state(samples)
        for j in range(nbr_steps):
            sample_j = {el: samples[el][:, j, :] if not el in self.not_sequential_data else samples[el] for el in
                        samples}
            output, state = func(sample_j, state, **kwargs)
            if first:
                first = False
                outputs = {el: tf.TensorArray(tf.float32, nbr_steps) for el in output}
            for el in outputs:
                outputs[el] = outputs[el].write(j, output[el])
        for el in outputs:
            outputs[el] = outputs[el].stack()
            outputs[el] = tf.transpose(outputs[el], tf.concat([[1, 0], tf.range(tf.size(outputs[el].shape[2:]))+2], axis = 0))

        return outputs, state

    def get_gradients(self, samples, max_bptt, func, variables_string, variables_func, compiled = True, complete_unroll = True, **kwargs):
        seq_length = samples['_lengths_mult'].shape[1]
        state = None
        j = 0
        grad_l = []
        loss_t = 0
        while j < seq_length:
            to_pass = {el: samples[el][:, j:j + max_bptt, :] if not el in self.not_sequential_data
                                                             else samples[el] for el in samples}
            with tf.GradientTape() as tape:
                if compiled:
                    if not complete_unroll:
                        outputs, state = self.forward_n(to_pass, state, func, tf.shape(to_pass['_lengths_mult'])[1], **kwargs)
                    else:
                        outputs, state = self.complete_unroll_forward_n(to_pass, state, func, to_pass['_lengths_mult'].shape[1], **kwargs)
                else:
                    outputs, state = self.dynamic_forward_n(to_pass, state, func, to_pass['_lengths_mult'].shape[1], **kwargs)
                output_sum = tf.add_n([outputs[el] for el in outputs])
                losses = output_sum * to_pass['_lengths_mult']
                loss_per_traj = tf.reduce_sum(losses, axis = 1)
                corrected_loss_per_traj = loss_per_traj / to_pass['_lengths']
                loss = tf.reduce_mean(corrected_loss_per_traj)
            to_train_vars = self.variables
            if not (variables_string is None):
                to_train_vars = sum([self.variables_d[el] for el in variables_string], [])
            if not (variables_func is None):
                to_train_vars = variables_func()
            current_gradient = tape.gradient(loss, to_train_vars)
            loss_t += loss
            grad_l.append(current_gradient)
            j += max_bptt
        with self.train_sum.as_default():
            tf.summary.scalar(func.__name__ + "/_total", loss_t, step=self.to_pickle['batch_nbr'])
        to_return_grad = [tf.add_n(el) if not el[0] is None else None for el in list(map(list, zip(*grad_l)))]

        self.to_pickle['batch_nbr'] += 1

        return self.post_process_gradients(to_return_grad)

    def post_process_gradients(self, grads_list):
        return grads_list

    def process_sample(self, sample):
        return sample

    def evaluate(self, input_data, func, compiled = True, batch_size = 50, complete_unroll = True, process_function = None, **kwargs):
        if process_function is None:
            process_function = self.process_sample
        dataset = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size).map(process_function).prefetch(tf.data.experimental.AUTOTUNE)
        state = None
        first = True
        loss = {}
        c = 1
        for cnt, samples in enumerate(dataset):
            if compiled:
                if not complete_unroll:
                    outputs, state = self.forward_n(samples, state, func, tf.shape(samples['_lengths_mult'])[1], **kwargs)
                else:
                    outputs, state = self.complete_unroll_forward_n(samples, state, func,
                                                                    samples['_lengths_mult'].shape[1], **kwargs)
            else:
                outputs, _ = self.dynamic_forward_n(samples, state, func, samples['_lengths_mult'].shape[1], **kwargs)
            cur_losses = {}
            for el in outputs:
                outputs[el] = outputs[el] * samples['_lengths_mult']
                cur_losses[el] = tf.reduce_sum(outputs[el], axis = 1)
                cur_losses[el] = cur_losses[el] / samples['_lengths']
                cur_losses[el] = tf.reduce_mean(cur_losses[el])
            cur_losses['_total'] = tf.add_n([cur_losses[el] for el in cur_losses])
            if first:
                loss = {el:cur_losses[el] for el in cur_losses}
                first = False
            else:
                for el in loss:
                    loss[el] += cur_losses[el]
            c += 1
        for el in loss:
            loss[el] = loss[el]/c
        with self.test_sum.as_default():
            for el in loss:
                tf.summary.scalar(func.__name__ + "/" + el, loss[el], step = self.to_pickle['batch_nbr'])
        return loss

    def train(self, input_data, func, max_bptt = 1, steps = 100, batch_size = 50, max_shuffle_buffer = 1000,
              variables_string = None, variables_func = None, epoch_mode = False, val_data = None, compiled = True, checkpoint_every = 1000,
              display_init = 100, complete_unroll = True, process_function = None, val_func = None, **kwargs):
        if val_func is None:
            val_func = func
        if process_function is None:
            process_function = self.process_sample
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.shuffle(max_shuffle_buffer).batch(batch_size, drop_remainder=True).map(process_function)
        dataset = dataset.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if epoch_mode:
            display = 1
        else:
            display = display_init
        old = time.time()
        for cnt, samples in enumerate(dataset):
            if epoch_mode:
                cnt_c = (cnt * batch_size) // input_data['_lengths'].shape[0]
            else:
                cnt_c = cnt

            if cnt_c > steps:
                break
            if cnt_c % display == 0:
                new = time.time()
                print("Training iteration .............", cnt_c, " ..... average iteration time = ", (new-old)/display)
                old = new
                if not val_data is None:
                    test_loss = self.evaluate(val_data, val_func, compiled, batch_size, complete_unroll=complete_unroll,
                                              process_function=process_function)
                    print ("Testing loss : ", test_loss)
                if cnt_c == display * 10:
                    display *= 10

            grads = self.get_gradients(samples, max_bptt, func, variables_string, variables_func, compiled = compiled, complete_unroll=complete_unroll, **kwargs)
            to_train_vars = self.variables
            if not (variables_string is None):
                to_train_vars = sum([self.variables_d[el] for el in variables_string], [])
            if not (variables_func is None):
                to_train_vars = variables_func()
            self.opt.apply_gradients(zip(grads, to_train_vars))

            if (cnt+1) % checkpoint_every == 0:
                self.store_model()

    def __call__(self, input_data, func, batch_size = 50, compiled = True, complete_unroll = True, process_function = None, **kwargs):
        if process_function is None:
            process_function = self.process_sample
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(batch_size, drop_remainder=False).map(process_function).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        first = True
        state = None
        whole_outputs = {}
        for samples in dataset:
            if compiled:
                if not complete_unroll:
                    outputs, _ = self.forward_n(samples, state, func, samples['_lengths_mult'].shape[1], **kwargs)
                else:
                    outputs, _ = self.complete_unroll_forward_n(samples, state, func, samples['_lengths_mult'].shape[1], **kwargs)
            else:
                outputs, _ = self.dynamic_forward_n(samples, state, func, samples['_lengths_mult'].shape[1], **kwargs)

            if first:
                whole_outputs = {el:[outputs[el]] for el in outputs}
                first = False
            else:
                for el in whole_outputs:
                    whole_outputs[el].append(outputs[el])

        whole_outputs = {el:tf.concat(whole_outputs[el], axis = 0) for el in whole_outputs}

        whole_outputs['_lengths'] = input_data['_lengths']
        whole_outputs['_lengths_mult'] = input_data['_lengths_mult']
        return whole_outputs