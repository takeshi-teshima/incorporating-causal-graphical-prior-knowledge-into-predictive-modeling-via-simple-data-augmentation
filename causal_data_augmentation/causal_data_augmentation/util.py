import itertools
import math


def build_batch(items, batch_size: int, item_count=None):
    '''Generates balanced baskets from iterable.
    Params:
        items : Iterable.
        batch_size : Maximum size of each batch. Surplus items are split evenly in the last two chunks.
        item_count : If `items` does not support `len()`, provide this parameter.

    Notes:
        Reference: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

    Examples:
        >>> from pprint import pprint
        >>> pprint(list(build_batch(list(range(11, 40)), 11)))
        [[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
         [22, 23, 24, 25, 26, 27, 28, 29, 30],
         [31, 32, 33, 34, 35, 36, 37, 38, 39]]

        >>> pprint(list(build_batch(list(range(0, 10)), 11)))
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        >>> pprint(list(build_batch(list(range(11, 41)), 10)))
        [[11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
         [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
         [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
    '''
    iterable = iter(items)
    item_count = item_count or len(items)

    is_last_chunk_uneven = (item_count % batch_size != 0)
    last_sum = batch_size + (item_count % batch_size)
    if is_last_chunk_uneven:
        last_sizes = [math.ceil(last_sum / 2), math.floor(last_sum / 2)]
    batch_count = 0
    while True:
        remaining_count = item_count - batch_count * batch_size

        if is_last_chunk_uneven and (remaining_count < 2 * batch_size) \
                and (remaining_count > batch_size):
            batch = list(itertools.islice(iterable, math.ceil(last_sum / 2)))
        elif is_last_chunk_uneven and (remaining_count < 2 * batch_size):
            batch = list(itertools.islice(iterable, math.floor(last_sum / 2)))
        else:
            batch = list(itertools.islice(iterable, batch_size))

        batch_count += 1

        if len(batch) > 0:
            yield batch
        else:
            break


if __name__ == '__main__':
    import doctest
    doctest.testmod()
