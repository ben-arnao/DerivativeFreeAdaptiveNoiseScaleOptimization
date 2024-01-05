import copy
from numpy.random import randn
import numpy as np


# User needs to provide the following:
# 1) a function 'get_model_params' that accepts 'model' as input and returns a flat array of floats as the output
# 2) a function 'set_model_params' that accepts 'model, params' as inputs, where params is a flat array of floats
# of the same size as the output from #1
# 3) a function 'get_model_score' that accepts 'model, inputs, outputs' as inputs, and return a float score as the output
# 4) an intialized model, along with inputs and outputs to fit the model to


def og_train(
        inputs,
        outputs,

        model,

        get_model_score,

        get_model_params,
        set_model_params,

        max_noise=1000,
        noise_adjust_momentum=0.99,
        init_noise_scale=0.01
):
    noise_scale = np.full_like(get_model_params(model), init_noise_scale)
    score_hist = [get_model_score(model, inputs, outputs)]

    while True:
        curr_params = copy.deepcopy(get_model_params(model))
        noise_arr, score_arr = [], []

        generate_noise = True
        i = 0

        # generate and score noise in weights
        while generate_noise:

            noise = np.random.normal(scale=noise_scale, size=len(curr_params))
            set_model_params(model, curr_params + noise)
            score = get_model_score(model, inputs, outputs)

            score_arr.append(score)
            noise_arr.append(noise)

            if score > max(score_hist):
                generate_noise = False

            i += 1
            if i >= max_noise:
                generate_noise = False

            if np.mean(score_arr) == 0:
                print('noise does not produce new score')
                return score_hist

        if max(score_arr) > max(score_hist):
            best_ind = score_arr.index(max(score_arr))
            step = noise_arr[best_ind]

            # adjust noise size relative to step size
            noise_scale = (noise_scale * noise_adjust_momentum) + (np.abs(step) * (1 - noise_adjust_momentum))
        else:
            print('did not find better score')
            return score_hist

        set_model_params(model, curr_params + step)  # take the step
        score_hist.append(get_model_score(model, inputs, outputs))

        print('\t\t\tstep: {}, score: {}, n_samples: {}'.format(len(score_hist) - 1, score_hist[-1], len(noise_arr)))
