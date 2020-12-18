import numpy as np
import time
import icpImpl
from path import Path
from tqdm import tqdm

# import pcl

# Constants
N = 10  # number of random points in the dataset
iterations = 50  # number of test iterations
dim = 3  # number of dimensions of the points
noise_sigma = .01  # standard deviation error to be added
translation = .1  # max translation of the test set
rotation = .1  # max rotation (radians) of the test set
rootdir = r'E:\data_odometry\dataset\sequences\09\velodyne_pcd'
dir='C:\\Users\\93121\\Desktop\\dataset\\'
numTests=10


def rotation_matrix(axis, theta):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.)
    b, c, d = -axis * np.sin(theta / 2.)

    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

def test_best_fit():
    # from prepare_data import loadPointCloud
    # pointClouds = loadPointCloud(rootdir)

    from prepare_data import loadPointCloud
    pointClouds = loadPointCloud(rootdir)

    total_time = 0
    for j in range(numTests):
        # for j in (range(len(pointClouds)-1)):
        A = np.asarray(pointClouds[j].points)
        B = np.asarray(pointClouds[j + 1].points)

        start = time.time()
        T, R1, t1 = icpImpl.best_fit_transform(B, A)
        total_time += time.time() - start
        print(T)

    print('best fit time: {:.3}'.format(total_time/numTests ))
    return


def test_icp():
    from prepare_data import loadPointCloud
    pointClouds = loadPointCloud(rootdir)

    total_time = 0
    f=open()
    abs_pose=np.identity(4,4)
    abs_pose=abs_pose[:2,:]
    print(abs_pose)
    # for j in range(len(pointclouds) - 1):
    for j in range(numTests):
        A = np.asarray(pointClouds[j].points)
        B = np.asarray(pointClouds[1].points)
        # Run ICP
        start = time.time()
        T, distances, iteration = icpImpl.icp(A, B, init_pose=None, max_iterations=iterations, tolerance=0.001)
        total_time += time.time() - start
        print('time: {:.3}'.format(time.time()-start)+' iters: {}'.format(iteration)+' distance: {}'.format(distances))
        print(T)
    print('icp time: {:.3}'.format(total_time / numTests))
    return
def test_icp_mock():
    from kitti_eval.VOLO_data_utils import test_framework_KITTI as test_framework
    dataset_dir = Path(dir)
    framework = test_framework(dataset_dir, '09', 3)
    for j, sample in enumerate(tqdm(framework)):
        # 加入LO优化
        pointclouds = sample['pointclouds']
        from VOLO import LO
        # pointcluds是可以直接处理的
        for i, pc in enumerate(pointclouds):
            if i == len(pointclouds) // 2:
                tgt_pc = pointclouds[i]

        optimized_transform_matrices = []
        for i, pc in enumerate(pointclouds):
            T, _, _ = LO(pc, tgt_pc, 50, T=None, LO='icp')
            optimized_transform_matrices.append(T)

        print('**********LO result*************')
        print(optimized_transform_matrices)


if __name__ == "__main__":
    #test_best_fit()
    test_icp()
    #test_icp_mock()
