import glob
import os
import argparse
from collections import defaultdict
import h5py
import numpy as np
import imageio

parser = argparse.ArgumentParser('script to generate training data')
parser.add_argument('--image', type=str,
                    default='D:/peoject/ffn/third_party/neuroproof_examples/training_sample2/grayscale_maps',
                    help='directory of images')
parser.add_argument('--label', type=str, default='./data/ffn/groundtruth.h5')
parser.add_argument('--save', type=str, default='data.h5', help='save file name')
parser.add_argument('--shape', type=list, default=[49, 49, 49], help='seed shape')
parser.add_argument('--thr', type=list, default=[0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
parser.add_argument('--min_size', type=int, default=10000)

args = parser.parse_args()


def _query_summed_volume(svt, diam):
    """Queries a summed volume table.
    Operates in 'VALID' mode, i.e. only computes the sums for voxels where the
    full diam // 2 context is available.
    Args:
    svt: summed volume table (see _summed_volume_table)
    diam: diameter (z, y, x tuple) of the area within which to compute sums
    Returns:
    sum of all values within a diam // 2 radius (under L1 metric) of every voxel
    in the array from which 'svt' was built.
    """
    return (
        svt[diam[0]:, diam[1]:, diam[2]:] - svt[diam[0]:, diam[1]:, :-diam[2]] -
        svt[diam[0]:, :-diam[1], diam[2]:] - svt[:-diam[0], diam[1]:, diam[2]:] +
        svt[:-diam[0], :-diam[1], diam[2]:] + svt[:-diam[0], diam[1]:, :-diam[2]]
        + svt[diam[0]:, :-diam[1], :-diam[2]] -
        svt[:-diam[0], :-diam[1], :-diam[2]])


def _summed_volume_table(val):
    """Computes a summed volume table of 'val'."""
    val = val.astype(np.int32)
    svt = val.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    return np.pad(svt, [[1, 0], [1, 0], [1, 0]], mode='constant')


def clear_dust(data, min_size=10):
    """Removes small objects from a segmentation array.
    Replaces objects smaller than `min_size` with 0 (background).
    Args:
    data: numpy array of segment IDs
    min_size: minimum size in voxels of an object to be retained
    Returns:
    the data array (modified in place)
    """
    ids, sizes = np.unique(data, return_counts=True)
    small = ids[sizes < min_size]
    small_mask = np.in1d(data.flat, small).reshape(data.shape)
    data[small_mask] = 0
    return data


def compute_partitions(seg_array, thresholds, lom_radius, min_size=10000):
    seg_array = clear_dust(seg_array, min_size=min_size)
    assert seg_array.ndim == 3

    lom_radius = np.array(lom_radius)
    lom_radius_zyx = lom_radius[::-1]
    lom_diam_zyx = 2 * lom_radius_zyx + 1

    def _sel(i):
        if i == 0:
            return slice(None)
        else:
            return slice(i, -i)

    valid_sel = [_sel(x) for x in lom_radius_zyx]
    output = np.zeros(seg_array[valid_sel].shape, dtype=np.uint8)
    corner = lom_radius

    labels = set(np.unique(seg_array))

    fov_volume = np.prod(lom_diam_zyx)
    for l in labels:
        # Don't create a mask for the background component.
        if l == 0:
            continue

    object_mask = (seg_array == l)

    svt = _summed_volume_table(object_mask)
    active_fraction = _query_summed_volume(svt, lom_diam_zyx) / fov_volume
    assert active_fraction.shape == output.shape

    # Drop context that is only necessary for computing the active fraction
    # (i.e. one LOM radius in every direction).
    object_mask = object_mask[valid_sel]

    # TODO(mjanusz): Use np.digitize here.
    for i, th in enumerate(thresholds):
        output[object_mask & (active_fraction < th) & (output == 0)] = i + 1

    output[object_mask & (active_fraction >= thresholds[-1]) & (output == 0)] = len(thresholds) + 1

    return corner, output


def run():
    png_files = glob.glob(os.path.join(args.image, '*.png'))
    png_files.sort()
    images = [imageio.imread(i) for i in png_files]
    images = np.array(images)
    with h5py.File(args.label, 'r') as f:
        label = f['stack'].value

    m = np.array([int(x/2) for x in args.shape])
    seg = label.copy()
    corner, partitions = compute_partitions(seg[...], [float(x) for x in args.thr], m, args.min_size)

    totals = defaultdict(int)  # partition -> voxel count
    indices = defaultdict(list)  # partition -> [(vol_id, 1d index)]
    vol_shapes = partitions.shape
    uniques, counts = np.unique(partitions, return_counts=True)
    for val, cnt in zip(uniques, counts):
        if val == 255:
            continue

        totals[val] += cnt
        indices[val].extend([flat_index for flat_index in np.flatnonzero(partitions == val)])

    max_count = max(totals.values())
    indices = np.concatenate([np.resize(np.random.permutation(v), max_count) for v in indices.values()], axis=0)
    np.random.shuffle(indices)
    coor = []
    for coord_idx in indices:
        z, y, x = np.unravel_index(coord_idx, vol_shapes)
        coor.append([x+m[0], y+m[1], z+m[2]])

    with h5py.File(args.save, 'w') as f:
        f.create_dataset('image', data=images, compression='gzip')
        f.create_dataset('label', data=label, compression='gzip')
        f.create_dataset('coor', data=coor, compression='gzip')


if __name__ == '__main__':
    run()
