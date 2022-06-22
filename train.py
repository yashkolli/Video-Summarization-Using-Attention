import numpy as np
import json
import os
from tqdm import tqdm, trange
import h5py
from prettytable import PrettyTable

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed, Concatenate, Lambda
from tensorflow.keras import Input, Model

from attention_layers.bahdanau_attention import BahdanauAttention
from attention_layers.luong_attention import LuongAttention
import eval


class BuildModel():

    def __init__(self, config=None, train_sequence_tvsum=None, train_sequence_summe=None, test_dataset_tvsum=None, test_dataset_summe=None):

        self.config = config
        self.train_sequence_tvsum = train_sequence_tvsum
        self.train_sequence_summe = train_sequence_summe
        self.test_dataset_tvsum = test_dataset_tvsum
        self.test_dataset_summe = test_dataset_summe

        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.acc_metric = tf.keras.metrics.BinaryAccuracy()

        if not os.path.exists(self.config.score_dir_tvsum):
            os.mkdir(self.config.score_dir_tvsum)

        if not os.path.exists(self.config.score_dir_summe):
            os.mkdir(self.config.score_dir_summe)

        if not os.path.exists(self.config.save_dir):
            os.mkdir(self.config.save_dir)

    @staticmethod
    def loss_fn(weights):
        def w_b_ce(y_true, y_pred):
            b_ce = K.binary_crossentropy(y_true, y_pred, from_logits=False)
            weight_vec = y_true * weights[1] + (1 - y_true) * weights[0]
            weighted_b_ce = weight_vec * b_ce
            return K.mean(weighted_b_ce)
        return w_b_ce

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.train_model(x, training=True)
            label_1 = tf.cast(tf.reduce_sum(y) / y.shape[0], tf.float32)
            label_0 = tf.cast(y.shape[1] - label_1, tf.float32)
            loss_function = self.loss_fn([label_1, label_0])
            loss_value = loss_function(y, logits)
        grads = tape.gradient(loss_value, self.train_model.trainable_weights)
        self.opt.apply_gradients(
            zip(grads, self.train_model.trainable_weights))
        self.acc_metric.update_state(y, logits)
        return loss_value

    # Basic Encoder-Decoder with Encoder Outputs as Inputs at each timestep
    @staticmethod
    def encoder_decoder_1():
        encoder_inputs = Input(shape=(320, 1024))

        encoder_BidirectionalLSTM = Bidirectional(
            LSTM(128, return_sequences=True, return_state=True))
        encoder_outputs, fh, fc, bh, bc = encoder_BidirectionalLSTM(
            encoder_inputs)
        ch = Concatenate()([fh, bh])
        cc = Concatenate()([fc, bc])
        encoder_states = [ch, cc]

        decoder_LSTM = LSTM(256, return_sequences=True)
        decoder_outputs = decoder_LSTM(
            encoder_outputs, initial_state=encoder_states)

        dense = TimeDistributed(Dense(1, activation='sigmoid'))
        decoder_outputs = dense(decoder_outputs)

        model = Model(encoder_inputs, decoder_outputs)
        return model

    # Encoder-Decoder with Decoder Outputs being fed as inputs for the next timestep
    @staticmethod
    def encoder_decoder_2():
        encoder_inputs = Input(shape=(320, 1024))

        encoder_BidirectionalLSTM = Bidirectional(LSTM(128, return_state=True))
        encoder_outputs, fh, fc, bh, bc = encoder_BidirectionalLSTM(
            encoder_inputs)
        ch = Concatenate()([fh, bh])
        cc = Concatenate()([fc, bc])
        encoder_states = [ch, cc]

        decoder_inputs = Input(shape=(1, 1))

        decoder_LSTM = LSTM(256, return_sequences=True, return_state=True)
        decoder_dense = Dense(1, activation='sigmoid')

        all_outputs = []

        inputs = decoder_inputs
        states = encoder_states
        for _ in range(320):
            outputs, sh, sc = decoder_LSTM(inputs, initial_state=states)
            outputs = decoder_dense(outputs)
            all_outputs.append(outputs)
            inputs = outputs
            states = [sh, sc]

        decoder_outputs = Lambda(
            lambda x: K.concatenate(x, axis=1))(all_outputs)

        train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        infenc_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(256, ))
        decoder_state_input_c = Input(shape=(256, ))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_LSTM(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)

        infdec_model = Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        return train_model, infenc_model, infdec_model

    # teacher-forcing
    @staticmethod
    def encoder_decoder_3():
        encoder_inputs = Input(shape=(320, 1024))

        encoder_BidirectionalLSTM = Bidirectional(LSTM(128, return_state=True))
        encoder_outputs, fh, fc, bh, bc = encoder_BidirectionalLSTM(
            encoder_inputs)
        ch = Concatenate()([fh, bh])
        cc = Concatenate()([fc, bc])
        encoder_states = [ch, cc]

        decoder_inputs = Input(shape=(None, 1))

        decoder_LSTM = LSTM(256, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_LSTM(
            decoder_inputs, initial_state=encoder_states)

        decoder_dense = Dense(1, activation='sigmoid')
        decoder_outputs = decoder_dense(decoder_outputs)

        train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        infenc_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(256, ))
        decoder_state_input_c = Input(shape=(256, ))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, sh, sc = decoder_LSTM(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [sh, sc]
        decoder_outputs = decoder_dense(decoder_outputs)

        infdec_model = Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        return train_model, infenc_model, infdec_model

    # Attentive Encoder-Decoder (Bahdanau)
    @staticmethod
    def encoder_decoder_4():
        encoder_inputs = Input(shape=(320, 1024))

        encoder_BidirectionalLSTM = Bidirectional(
            LSTM(128, return_sequences=True, return_state=True))
        encoder_outputs, fh, fc, bh, bc = encoder_BidirectionalLSTM(
            encoder_inputs)
        ch = Concatenate()([fh, bh])
        cc = Concatenate()([fc, bc])
        encoder_states = [ch, cc]

        attention = BahdanauAttention(256)

        decoder_inputs = Input(shape=(1, 1))
        decoder_LSTM = LSTM(256, return_sequences=True,
                            return_state=True, name="here")
        decoder_dense = Dense(1, activation='sigmoid')

        all_outputs = []

        inputs = decoder_inputs
        decoder_outputs = tf.expand_dims(ch, 1)
        states = encoder_states

        for _ in range(320):
            context_vector, attention_weights = attention(
                decoder_outputs, encoder_outputs)
            inputs = tf.concat([context_vector, inputs], axis=-1)
            decoder_outputs, sh, sc = decoder_LSTM(
                inputs, initial_state=states)
            outputs = decoder_dense(decoder_outputs)
            all_outputs.append(outputs)

            inputs = outputs
            states = [sh, sc]

        decoder_outputs = Lambda(
            lambda x: K.concatenate(x, axis=1))(all_outputs)

        train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        infenc_model = Model(
            encoder_inputs, [encoder_outputs] + encoder_states)

        attention_inputs_1 = Input(shape=(1, 256,))
        attention_inputs_2 = Input(shape=(320, 256,))
        cv, aw = attention(query=attention_inputs_1, value=attention_inputs_2)
        attention_outputs = [cv, aw]

        attention_model = Model(
            [attention_inputs_1, attention_inputs_2], attention_outputs)

        decoder_inputs_inference = Input(shape=(1, (1 + 256)))
        decoder_state_input_h = Input(shape=(256, ))
        decoder_state_input_c = Input(shape=(256, ))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, sh, sc = decoder_LSTM(
            decoder_inputs_inference, initial_state=decoder_states_inputs)
        decoder_states = [sh, sc]
        decoder_dense_outputs = decoder_dense(decoder_outputs)

        infdec_model = Model([decoder_inputs_inference] + decoder_states_inputs,
                             [decoder_outputs, decoder_dense_outputs] + decoder_states)

        return train_model, infenc_model, attention_model, infdec_model

    # Attentive Encoder-Decoder (Luong)
    @staticmethod
    def encoder_decoder_5():
        encoder_inputs = Input(shape=(320, 1024))

        encoder_BidirectionalLSTM = Bidirectional(
            LSTM(128, return_sequences=True, return_state=True))
        encoder_outputs, fh, fc, bh, bc = encoder_BidirectionalLSTM(
            encoder_inputs)
        ch = Concatenate()([fh, bh])
        cc = Concatenate()([fc, bc])
        encoder_states = [ch, cc]

        attention = LuongAttention(256)
        attention_dense = Dense(256, activation=tf.math.tanh)

        decoder_inputs = Input(shape=(1, 1))
        decoder_LSTM = LSTM(256, return_sequences=True, return_state=True)
        decoder_dense = Dense(1, activation='sigmoid')

        all_outputs = []

        inputs = decoder_inputs
        states = encoder_states

        for _ in range(320):
            decoder_outputs, sh, sc = decoder_LSTM(
                inputs, initial_state=states)
            context_vector, attention_weights = attention(
                query=decoder_outputs, value=encoder_outputs)
            context_and_decoder_outputs = tf.concat(
                [context_vector, decoder_outputs], axis=-1)
            attention_vector = attention_dense(context_and_decoder_outputs)
            outputs = decoder_dense(attention_vector)
            all_outputs.append(outputs)

            inputs = outputs
            states = [sh, sc]

        decoder_outputs = Lambda(
            lambda x: K.concatenate(x, axis=1))(all_outputs)

        train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        infenc_model = Model(
            encoder_inputs, [encoder_outputs] + encoder_states)

        decoder_state_input_h = Input(shape=(256, ))
        decoder_state_input_c = Input(shape=(256, ))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, sh, sc = decoder_LSTM(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [sh, sc]

        infdec_model = Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        attention_inputs_1 = Input(shape=(1, 256,))
        attention_inputs_2 = Input(shape=(320, 256,))
        cv, aw = attention(query=attention_inputs_1, value=attention_inputs_2)
        attention_outputs = [cv, aw]

        attention_model = Model(
            [attention_inputs_1, attention_inputs_2], attention_outputs)

        attention_dense_inputs = Input(shape=(1, 512))
        attention_vector = attention_dense(attention_dense_inputs)
        decoder_dense_outputs = decoder_dense(attention_vector)

        decoder_dense_model = Model(
            attention_dense_inputs, decoder_dense_outputs)

        return train_model, infenc_model, infdec_model, attention_model, decoder_dense_model

    @tf.function
    def predict_fn_1(self, features):
        state = self.infenc_model(features)
        target_sequence = np.array([2], dtype="float32").reshape(1, 1, 1)
        # output = []
        output = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        for t in tf.range(320):
            decoder_output, sh, sc = self.infdec_model(
                [target_sequence] + state)
            # output.append(decoder_output[0, 0, :])
            output = output.write(t, decoder_output[0, 0, :])

            state = [sh, sc]
            target_seqeunce = decoder_output

        # return np.array(output)
        return output.stack()

    @tf.function
    def predict_fn_2(self, features):
        encoder_outputs, sh, sc = self.infenc_model(features)
        decoder_outputs = tf.expand_dims(sh, 1)
        target_sequence = np.array([2], dtype="float32").reshape(1, 1, 1)
        states = [sh, sc]
        # output = []
        output = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        for t in tf.range(320):
            cv, aw = self.attention_model([decoder_outputs, encoder_outputs])
            target_sequence = tf.concat([cv, target_sequence], axis=-1)
            decoder_outputs, decoder_dense_outputs, sh, sc = self.infdec_model(
                [target_sequence] + states)
            # output.append(decoder_dense_outputs[0, 0, :])
            output = output.write(t, decoder_dense_outputs[0, 0, :])

            states = [sh, sc]
            target_sequence = decoder_dense_outputs

        # return np.array(output)
        return output.stack()

    @tf.function
    def predict_fn_3(self, features):
        encoder_outputs, sh, sc = self.infenc_model(features)
        target_sequence = np.array([2], dtype="float32").reshape(1, 1, 1)
        states = [sh, sc]
        # output = []
        output = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        for t in tf.range(320):
            decoder_outputs, sh, sc = self.infdec_model(
                [target_sequence] + states)
            context_vector, attention_weights = self.attention_model(
                [decoder_outputs, encoder_outputs])
            context_and_decoder_outputs = tf.concat(
                [context_vector, decoder_outputs], axis=-1)
            outputs = self.decoder_dense_model(context_and_decoder_outputs)
            # output.append(outputs[0, 0, :])
            output = output.write(t, outputs[0, 0, :])

            target_sequence = outputs
            states = [sh, sc]

        # return np.array(output)
        return output.stack()

    def train(self):
        # self.train_model = self.encoder_decoder_1()
        # self.train_model, self.infenc_model, self.infdec_model = self.encoder_decoder_2()
        # self.train_model, self.infenc_model, self.infdec_model = self.encoder_decoder_3()
        # self.train_model, self.infenc_model, self.attention_model, self.infdec_model = self.encoder_decoder_4()
        self.train_model, self.infenc_model, self.infdec_model, self.attention_model, self.decoder_dense_model = self.encoder_decoder_5()

        t = trange(self.config.n_epochs, desc='Epoch', ncols=90)
        for epoch_i in t:

            """for model - 1"""
            # for batch_i, (feature, label) in enumerate(tqdm(self.train_sequence_tvsum, desc = "Batch_TVSum", ncols = 80, leave = False)):
            #   loss_value = self.train_step(feature, label)

            # for batch_i, (feature, label) in enumerate(tqdm(self.train_sequence_summe, desc = "Batch_Summe", ncols = 80, leave = False)):
            #     loss_value = self.train_step(feature, label)

            """for model - 3"""
            # for batch_i, ([feature, shifted_label], label) in enumerate(tqdm(self.train_sequence_tvsum, desc = "Batch_TVSum", ncols = 80, leave = False)):
            #   loss_value = self.train_step([feature, shifted_label], label)

            # for batch_i, ([feature, shifted_label], label) in enumerate(tqdm(self.train_sequence_summe, desc = "Batch_Summe", ncols = 80, leave = False)):
            #     loss_value = self.train_step([feature, shifted_label], label)

            """for model - 2, 4, 5"""

            for batch_i, ([feature, sdata], label) in enumerate(tqdm(self.train_sequence_tvsum, desc="Batch_TVSum", ncols=80, leave=False)):
                loss_value = self.train_step([feature, sdata], label)

            # for batch_i, ([feature, sdata], label) in enumerate(tqdm(self.train_sequence_summe, desc = "Batch_Summe", ncols = 80, leave = False)):
            #     loss_value = self.train_step([feature, sdata], label)

            train_acc = self.acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            print("loss:  ", loss_value)
            self.acc_metric.reset_states()

            ckpt_path = self.config.save_dir + '/epoch-{}.ckpt'.format(epoch_i)
            tqdm.write("Save parameters at {}".format(ckpt_path))
            self.train_model.save_weights(ckpt_path)
            self.evaluate('tvsum', epoch_i)
            # self.evaluate('summe', epoch_i)

    def evaluate(self, dataset, epoch_i):

        out_dict = {}
        eval_arr = []
        table = PrettyTable()
        table.title = 'Evaluation Result of epoch {}'.format(epoch_i)
        table.field_names = ['ID', 'Precision', 'Recall', 'F-Score']
        table.float_format = '1.5'

        if dataset == "tvsum":
            data_path = self.config.data_path_tvsum
            test_dataset = self.test_dataset_tvsum
            score_dir = self.config.score_dir_tvsum

        elif dataset == 'summe':
            data_path = self.config.data_path_summe
            test_dataset = self.test_dataset_summe
            score_dir = self.config.score_dir_summe

        with h5py.File(data_path) as data_file:
            for feature, label, index in tqdm(test_dataset, desc='Evaluate', ncols=90, leave=False):

                """enocder-decoder - 1"""
                # pred_score = self.train_model.predict(feature.reshape(-1,320,1024))[0]
                """encoder-decoder - 2, 3"""
                # pred_score = self.predict_fn_1(feature.reshape(-1,320,1024))
                """encoder-decoder - 4"""
                # pred_score = self.predict_fn_2(feature.reshape(-1,320,1024))
                """encoder-decoder - 5"""
                pred_score = self.predict_fn_3(feature.reshape(-1, 320, 1024))

                video_info = data_file['video_'+str(index)]
                pred_score, pred_selected, pred_summary = eval.select_keyshots(
                    video_info, pred_score)
                true_summary_arr = video_info['user_summary'][:]
                eval_res = [eval.eval_metrics(
                    pred_summary, true_summary) for true_summary in true_summary_arr]
                eval_res = np.mean(eval_res, axis=0).tolist()

                eval_arr.append(eval_res)
                table.add_row([index] + eval_res)

                out_dict[str(index)] = {
                    'pred_score': pred_score,
                    'pred_selected': pred_selected,
                    'pred_summary': pred_summary
                }

        score_save_path = score_dir + '/epoch-{}.json'.format(epoch_i)
        with open(score_save_path, 'w') as f:
            tqdm.write('Save score at {}'.format(str(score_save_path)))
            json.dump(out_dict, f)
        if dataset == "tvsum":
            eval_mean = np.mean(eval_arr, axis=0).tolist()
            table.add_row(['mean'] + eval_mean)
        else:
            eval_max = np.amax(eval_arr, axis=0).tolist()
            table.add_row(['max'] + eval_max)
        tqdm.write(str(table))


if __name__ == '__main__':
    from config import Config
    from data_loader import get_loader
    train_config = Config()
    train_sequence_tvsum, test_dataset_tvsum = get_loader(
        train_config.data_path_tvsum, batch_size=train_config.batch_size)
    train_sequence_summe, test_dataset_summe = get_loader(
        train_config.data_path_summe, batch_size=train_config.batch_size)
    builder = BuildModel(train_config, train_sequence_tvsum,
                         train_sequence_summe, test_dataset_tvsum, test_dataset_summe)
    builder.train()
