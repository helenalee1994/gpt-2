# new version on interact
#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
    filename='',
    nrecipes=2*4*100, 
    overwrite=False
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    :filename='' : Path to the X_test set; each input is splited by \n\n
     nrecipes=2*4*100, sampling 100 sets of recipes; ignore the odd lines (*2), four fields(*4), 100 sets(*100)
    :overwrite=False : whether to overwrite the y_pred
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
        
        with open(filename, 'r') as f:
            to_write = ''
            for i, raw_text in enumerate(f):
                if i < nrecipes:
                    if not i%2: 
                        context_tokens = enc.encode(raw_text)
                        # may be useful if we want to evaluate the fields respectively
                        #last_token = raw_text.split(' ')[-1].replace('\n','')

                        generated = 0
                        for _ in range(nsamples // batch_size):
                            out = sess.run(output, feed_dict={
                                context: [context_tokens for _ in range(batch_size)]
                            })[:, len(context_tokens):]

                            for i in range(batch_size):
                                generated += 1
                                text = enc.decode(out[i])
                                # not interested in the words after '<'
                                text = text.split('<')[0] 

                    # filter out \n only sentences
                    else:
                        text = '\n'

                    # show some progress    
                    if (i/2/4)%10 == 0:
                        print('processing file %d ' % (i/2/4))
                    to_write += text
                else:
                    break
        save(filename.replace('X_test','y_pred'), to_write, overwrite)
        
def save(filename, to_write, overwrite = False):
    make_dir(filename)
    if os.path.isfile(filename) == True and overwrite == False:
        print('already exists'+filename)
    else:    
        with open(filename,'w') as f:
            f.write('%s' % to_write)
        print('saved '+filename)
        
def make_dir(filename):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('make dir')
            
if __name__ == '__main__':
    fire.Fire(interact_model)
