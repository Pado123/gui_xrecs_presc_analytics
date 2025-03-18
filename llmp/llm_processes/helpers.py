import torch
import math
import re
import decimal
import numpy as np
import sys


ctx = decimal.Context()
ctx.prec = 20


def _map_to_ordinal(array, ordering):
    if ordering is not None:
        return np.array([ordering[key] for key in array])
    else:
        return array


def scale_y(ys, old_min, old_max, new_min, new_max):
    assert ys.ndim == 1
    return ((ys - old_min) * (new_max - new_min) / (old_max - old_min)) + new_min


def randomize(x, y):
    permutation = np.random.permutation(len(x))
    return  (np.array(x))[permutation], (np.array(y))[permutation]


def sequential_sort(x, y, x_ordering):
    sort_indices = np.argsort(np.array(_map_to_ordinal(x, x_ordering)))
    return (np.array(x))[sort_indices], (np.array(y))[sort_indices]


def sort_test_by_distance_from_train(x_train, x_test, y_test):
    dim_x = x_train.ndim
    distances = []
    for i in x_test:
        min_distance = sys.float_info.max
        for j in x_train: 
            if dim_x > 1:
                distance = math.dist(i, j)
            else:
                distance = abs(i - j)
            if distance < min_distance:
                min_distance = distance
        distances.append(min_distance)
    distances = np.array(distances)
    sort_indices = np.argsort(distances)
    x_test_sorted = (np.array(x_test))[sort_indices]
    y_test_sorted = (np.array(y_test))[sort_indices]

    return x_test_sorted, y_test_sorted


def get_dimension(a):
    if a.ndim > 1:
        return a.shape[1] # return the second dimension size
    else:
        return 1

def _float_to_str(f, num_decimal=None, add_spaces=False):
    """Convert float to string without resorting to scientific notation."""
    if isinstance(f, float):
        d1 = ctx.create_decimal_from_float(f)
    if isinstance(f, np.int64) or isinstance(f, np.int32):
        d1 = ctx.create_decimal(str(f))
    else:
        d1 = ctx.create_decimal(repr(f))
    if num_decimal is not None:
        d1 = round(d1, num_decimal)
    s = format(d1, 'f')
    if add_spaces:
        s = (" ".join(s))
    return s 


def floats_to_str(nums, num_decimal, dim=1, add_spaces=False):
    if np.ndim(nums) == 0:
        return _float_to_str(nums, num_decimal, add_spaces)  # when y_dim = 1, only a scalar is passed
    assert len(nums) > 0
    if dim > 1:  # can have multiple dimensions in x and y
        return [[_float_to_str(value, num_decimal, add_spaces) for value in group] for group in nums]
    else:
        return [_float_to_str(num, num_decimal, add_spaces) for num in nums]


def _format_observed_data_point(x, y, dim_x, dim_y, first_prefix, next_prefix, break_str):
    if (dim_x > 1) and (dim_y > 1):
        x_point_string = ''
        for i in range(dim_x):
            if i == 0:
                x_point_string += first_prefix
            else:
                x_point_string += next_prefix
            x_point_string += x[i]
        y_point_string = ''
        for i in range(dim_y):
            y_point_string += next_prefix
            y_point_string += y[i]
        return f'{x_point_string}{y_point_string}{break_str}'
    elif dim_x > 1:
        x_point_string = ''
        for i in range(dim_x):
            if i == 0:
                x_point_string += first_prefix
            else:
                x_point_string += next_prefix
            x_point_string += x[i]
        return f'{x_point_string}{next_prefix}{y}{break_str}'
    elif dim_y > 1:
        y_point_string = ''
        for i in range(dim_y):
            y_point_string += next_prefix
            y_point_string += y[i]
        return f'{first_prefix}{x}{y_point_string}{break_str}'
    else: # dim_x = dim_y = 1
       return f'{first_prefix}{x}{next_prefix}{y}{break_str}'
    

def _format_query_data_point(x, dim_x, first_prefix, next_prefix):
    if dim_x > 1:
        x_point_string = ''
        for i in range(dim_x):
            if i == 0:
                x_point_string += first_prefix
            else:
                x_point_string += next_prefix
            x_point_string += x[i]
        return f'{x_point_string}{next_prefix}'
    else: # dim_x = dim_y = 1
       return f'{first_prefix}{x}{next_prefix}'


def get_model_context_length(model):
    """Get the maximum context length from model config."""
    typical_fields = [
        "max_position_embeddings", 
        "n_positions", 
        "seq_len", 
        "seq_length", 
        "n_ctx", 
        # "sliding_window"
    ]
    
    context_windows = [
        getattr(model.config, field) 
        for field in typical_fields 
        if hasattr(model.config, field)
    ]
    
    if context_windows:
        return context_windows[-1]  # Get the last one as it's often the most relevant
    return None

def construct_prompts(
        x_train,
        y_train,
        x_test,
        train_frac=1.0,
        prefix='',
        x_prefix='',
        y_prefix=', ',
        break_str='\n',
        remove_space=True,
        dim_x=1,
        dim_y=1,
        num_decimal_x=0,
        num_decimal_y=0,
        order='distance',
        add_spaces=False,
        x_ordering=None,
        chat_template=None,
        tokenizer=None,
        model=None,
        max_new_tokens=128  # Reserve tokens for generation
        ):
    
    # Get model's context length if available
    max_context_length = None
    if model is not None:
        max_context_length = get_model_context_length(model)
    
    # Convert xy train and x test to str
    if x_ordering is not None:  # xs are already a string
        str_x_train = x_train
        str_x_test = x_test
    else:
        str_x_train = floats_to_str(x_train, num_decimal_x, dim_x, add_spaces)
        str_x_test = floats_to_str(x_test, num_decimal_x, dim_x, add_spaces)
    str_y_train = floats_to_str(y_train, num_decimal_y, dim_y, add_spaces)

    # Build training data points
    train_points = []
    for x, y in zip(str_x_train, str_y_train):
        point = _format_observed_data_point(
            x=x,
            y=y,
            dim_x=dim_x,
            dim_y=dim_y,
            first_prefix=x_prefix,
            next_prefix=y_prefix,
            break_str=break_str
        )
        train_points.append(point)

    prompts = []
    for (xt_str, xt_num) in zip(str_x_test, _map_to_ordinal(x_test, x_ordering)):
        if order == 'distance':
            distances = []
            for value in _map_to_ordinal(x_train, x_ordering):
                if dim_x > 1:
                    distances.append(math.dist(xt_num, value))
                else:
                    distances.append(abs(xt_num - value))
            sort_indices = np.flip(np.argsort(distances))
            ordered_points = [train_points[i] for i in sort_indices]
        elif order == 'random':
            ordered_points = train_points
        elif order == 'sequential': 
            raise NotImplementedError # TODO FIX THIS making sure it words with some sort of ordering in PM
            sort_indices = np.argsort(np.array(_map_to_ordinal(x_train, x_ordering)))
            ordered_points = [train_points[i] for i in sort_indices]

        # Format test point
        test_point = _format_query_data_point(
            x=xt_str,
            dim_x=dim_x,
            first_prefix=x_prefix,
            next_prefix=y_prefix
        )

        if chat_template and tokenizer:
            # Format as chat messages
            messages = [
                {"role": "system", "content": "You are a process mining expert. You can predict the lead time, that is the total time that takes from start to end an activity, from process logs."},
                {"role": "user", "content": prefix} if prefix else None,
                {"role": "assistant", "content": ""},
            ]
            base_prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, tokenize=False)
            base_tokens = len(tokenizer.encode(base_prompt))
            
            # Calculate remaining tokens for training points
            if max_context_length:
                available_tokens = max_context_length - base_tokens - max_new_tokens
                available_tokens *= train_frac
                
                # Add training points until we reach the token limit
                final_points = []
                current_tokens = 0
                for point in ordered_points:
                    point_tokens = len(tokenizer.encode(point))
                    if current_tokens + point_tokens > available_tokens:
                        break
                    final_points.append(point)
                    current_tokens += point_tokens 
                
                # Add test point
                test_tokens = len(tokenizer.encode(test_point))
                if current_tokens + test_tokens <= available_tokens:
                    prompt = base_prompt + "".join(final_points) + test_point
                else:
                    # If test point doesn't fit, remove last training point
                    if final_points:
                        final_points.pop()
                    prompt = base_prompt + "".join(final_points) + test_point
            else:
                # If no context length found, use all points
                prompt = base_prompt + "".join(ordered_points) + test_point
        else:
            # Traditional prompt format
            if max_context_length and tokenizer:
                base_tokens = len(tokenizer.encode(prefix))
                available_tokens = max_context_length - base_tokens - max_new_tokens
                
                final_points = []
                current_tokens = 0
                for point in ordered_points:
                    point_tokens = len(tokenizer.encode(point))
                    if current_tokens + point_tokens > available_tokens:
                        break
                    final_points.append(point)
                    current_tokens += point_tokens
                
                test_tokens = len(tokenizer.encode(test_point))
                if current_tokens + test_tokens <= available_tokens:
                    prompt = prefix + "".join(final_points) + test_point
                else:
                    if final_points:
                        final_points.pop()
                    prompt = prefix + "".join(final_points) + test_point
            else:
                prompt = prefix + "".join(ordered_points) + test_point
            
        if remove_space:
            prompt = prompt.rstrip(' ')
        prompts.append(prompt)
    
    if max_context_length and tokenizer:
        # Check if any prompts exceed the context length
        for prompt in prompts:
            if len(tokenizer.encode(prompt)) > max_context_length:
                raise ValueError("Prompt exceeds model's maximum context length.")
    
    # Print the number of training points
    print(f"The number of training points is {len(final_points)} out of {len(ordered_points)} ({len(final_points)/len(ordered_points) * 100:.2f} %)")
    
    return prompts

def _generate_max_min_values(n, k):
    # Calculate the part before the decimal
    before_decimal = sum(9 * 10**i for i in range(n))
    # Calculate the part after the decimal
    after_decimal = sum(9 * 10**-i for i in range(1, k + 1))
    # Combine both parts
    return before_decimal + after_decimal

def get_num_from_gen(gen, break_str='\n', dim_y=1, max_generated_length=7, num_decimal_places_y=2):
    gen = gen.replace(" ","") # remove any spaces, we add spaces for phi
    nums = re.findall(r'-?\d+\.?\d*', gen)
    
    # if the generataion does not contain any numbers, return None, throw away sample
    if not nums:
        return None
    
    if dim_y > 1:
        # throw away sample if it doesn't contain a break_str
        if break_str not in gen:
             return None
        assert len(nums) >= dim_y
        res = []
        for i in range(dim_y):
            res.append(float(nums[i]))
        res = np.array(res)
    else:
        # determine max and min generated values 
        if num_decimal_places_y == 0:
            max_val = _generate_max_min_values(max_generated_length - 1, 0)
            min_val = -_generate_max_min_values(max_generated_length - 2, 0)
        else:
            max_val = _generate_max_min_values(max_generated_length - num_decimal_places_y - 2, num_decimal_places_y)
            min_val = -_generate_max_min_values(max_generated_length - num_decimal_places_y - 3, num_decimal_places_y)
        
        res = float(nums[0])
        
        if break_str not in gen and "." not in nums[0]:
            if res > max_val:
                res = max_val
            elif res < min_val:    
                res = min_val
            else:
                res = None        
        
    return res


def compute_mse(a, b):
    return np.mean((np.array(a) - np.array(b)) ** 2)


def process_generated_results(gen_results, break_str='\n', dim_y=1, max_generated_length=7, num_decimal_places_y=2):
    # Get all sampled y values. Shape is (num ys, num samples).
    num_xs = len(gen_results['data']['x_test'])
    y_tests = [[] for _ in range(num_xs)]
    y_test_mean = [np.nan for _ in range(num_xs)]
    y_test_median = [np.nan for _ in range(num_xs)]
    y_test_std = [np.nan for _ in range(num_xs)]
    y_test_lower = [np.nan for _ in range(num_xs)]
    y_test_upper = [np.nan for _ in range(num_xs)]
    for i in range(len(gen_results['gen'])):
        if not gen_results['gen'][i]:
            continue
        ys = []
        for j, txt in enumerate(gen_results['gen'][i]):
            y = get_num_from_gen(
                gen=txt,
                break_str=break_str,
                dim_y=dim_y,
                max_generated_length=max_generated_length,
                num_decimal_places_y=num_decimal_places_y
            )
            if y is not None:
                ys.append(y)
        y_tests[i] += ys
        if dim_y > 1:
            ys = np.array(ys)
            y_test_mean[i] = np.mean(ys, axis=0)
            y_test_median[i] = np.median(ys, axis=0)
            y_test_std[i] = np.std(ys, axis=0)
            y_test_lower[i] = np.percentile(ys, 2.5, axis=0)
            y_test_upper[i] = np.percentile(ys, 97.5, axis=0)  
        else:
            y_test_mean[i] = np.mean(ys)
            y_test_median[i] = np.median(ys)
            y_test_std[i] = np.std(ys)
            y_test_lower[i] = np.percentile(ys, 2.5)
            y_test_upper[i] = np.percentile(ys, 97.5)

    if dim_y > 1:
        mae = [np.mean(np.abs((np.array(y_test_median)[:, i] -
                               gen_results['data']['y_test'][:, i]))) 
               for i in range(gen_results['data']['y_test'].shape[1])]
        
        mse = [compute_mse(np.array(y_test_mean)[:, i], 
                           gen_results['data']['y_test'][:, i]) 
               for i in range(gen_results['data']['y_test'].shape[1])]
        
    else:
        mse = compute_mse(
            y_test_mean,
            gen_results['data']['y_test'][: len(y_test_mean)]
        )

        mae = np.mean(np.abs(y_test_mean - np.array(gen_results['data']['y_test'])))
        mad = np.mean(np.abs(y_test_median - np.array(gen_results['data']['y_test'])))


    gen_results['y_test'] = y_tests
    if dim_y == 1:  # only used in black box opt with one output y
        gen_results['y_test_max_x'] = gen_results['data']['x_test'][np.argmax(np.max(np.array(y_tests), axis=1))]  # find argmax of the largest sample
    gen_results['y_test_mean'] = y_test_mean
    gen_results['y_test_median'] = y_test_median
    gen_results['y_test_std'] = y_test_std
    gen_results['y_test_lower'] = y_test_lower
    gen_results['y_test_upper'] = y_test_upper
    gen_results['mse'] = mse
    gen_results['mae'] = mae
    gen_results['mad'] = mad

    print(f'mae: {mae}')
    print(f'mse: {mse}')
    print(f'mad: {mad}')
    return gen_results


# this API is in python 3.9, but this is need if running python < 3.9
def my_removesuffix(self: str, suffix: str, /) -> str:
    # suffix='' should not call self[:-0].
    if suffix and self.endswith(suffix):
        return self[:-len(suffix)]
    else:
        return self[:]
