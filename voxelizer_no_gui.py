import os
import numpy as np
import copy

data_dir = '/home/joey/dataset/Data/SemanticKITTI/sequences/00/'

current_frame = 10

prior_frame = 10
past_frame = 10

def read_all_pose(begin_frame, current_frame, last_frame, data_dir, dataset):
    if dataset == 'semantic_kitti':
        with open(data_dir + 'poses.txt') as fd:
            lines = [i.split(' ') for i in fd.read().splitlines()]
        poses = lines[begin_frame:last_frame]
        poses_np = np.array(poses, dtype=np.float32)
        row = poses_np.shape[0]
        to_append = np.broadcast_to(np.array([0., 0., 0., 1.]), (row, 4))
        matrices = np.hstack((poses_np, to_append))
        return matrices[:current_frame - begin_frame, :], matrices[current_frame-begin_frame:current_frame-begin_frame+1, :], matrices[current_frame-begin_frame+1:, :]
    
def read_pc(begin_frame, current_frame, last_frame, data_dir, dataset):
    if dataset == 'semantic_kitti':
        def load_pc(start, end):
            result = []
            for i in range(start, end):
                filename = '/velodyne/' + str(i).zfill(6)+'.bin'
                pc = np.fromfile(data_dir + filename, dtype=np.float32).reshape(-1, 4)
                result.append(pc)
            return result
        past_pc = load_pc(begin_frame, current_frame)
        current_pc = load_pc(current_frame, current_frame + 1)
        prior_pc = load_pc(current_frame + 1, last_frame)
        return past_pc, current_pc, prior_pc

def read_labels(begin_frame, current_frame, last_frame, data_dir, dataset):
    if dataset == 'semantic_kitti':
        def load_label(start, end):
            result = []
            for i in range(start, end):
                filename = '/labels/' + str(i).zfill(6)+'.label'
                label = np.fromfile(data_dir + filename, dtype=np.int32).reshape(-1, 1)
                result.append(label)
            return result
        past_label = load_label(begin_frame, current_frame)
        current_label = load_label(current_frame, current_frame + 1)
        prior_label = load_label(current_frame + 1, last_frame)
        return past_label, current_label, prior_label

def transform_all_pcs(poses, all_pcs, all_labels):
    def get_transformation(src, tgt):
        T_src = src
        R_src = T_src[:3, :3]
        t_src = T_src[:3, 3]
        T_src_inv = np.zeros((4,4))
        T_src_inv[:3, :3] = R_src.T
        T_src_inv[:3, 3] = -np.matmul(R_src.T, t_src)
        T_src_inv[3,3] = 1
        T_src_tgt = np.matmul(tgt, T_src_inv)
        return T_src_tgt

    def transform_all_points(poses, T_tgt, pcs, labels):
        result_pcs = np.empty((0,4), dtype=np.float32)
        result_label = np.empty((0,1), dtype=np.int32)
    
        for idx, pose in enumerate(poses):
            T_src = np.array(pose).reshape(4, 4)
            T_tgt_src = get_transformation(T_tgt, T_src)
            old_pc = pcs[idx]
            new_pc = np.ones_like(old_pc)
            new_pc[:,:3] = np.copy(old_pc[:,:3])
            result_pc = (T_tgt_src @ new_pc.T).T
            result_pc[:,3] = old_pc[:,3]
            result_pcs = np.concatenate((result_pcs, np.copy(result_pc)), axis = 0)    
            result_label = np.concatenate((result_label, labels[idx]), axis=0)
        return result_pcs, result_label 

    past_src_pose, target_pose, prior_src_pose = poses
    past_src_pc, target_pc, prior_src_pc = all_pcs
    past_label, target_label, prior_label = all_labels
    
    T_tgt = np.array(target_pose).reshape(4, 4)
    past_pcs, past_labels = transform_all_points(past_src_pose, T_tgt, past_src_pc, past_label)
    prior_pcs, prior_labels = transform_all_points(prior_src_pose, T_tgt, prior_src_pc, prior_label)    
    final_result_pc = np.concatenate((past_pcs, target_pc[0], prior_pcs), axis=0)
    final_result_label = np.concatenate((past_labels, target_label[0], prior_labels), axis=0)

    return final_result_pc, final_result_label

if __name__ == '__main__':
    print('start processing frame: ', current_frame)
    for tmp in os.walk(data_dir + 'velodyne/'):
        fds = tmp[2]
    begin_frame = max(0, current_frame - prior_frame)
    last_frame = min(current_frame + past_frame + 1, len(fds))
    poses = read_all_pose(begin_frame, current_frame, last_frame, data_dir, 'semantic_kitti')
    all_pcs = read_pc(begin_frame, current_frame, last_frame, data_dir, 'semantic_kitti')
    all_labels = read_labels(begin_frame, current_frame, last_frame, data_dir, 'semantic_kitti')
    cvted_pcs, cvted_labels = transform_all_pcs(poses, all_pcs, all_labels)
    import pdb; pdb.set_trace()
    print('finish processing frame: ', current_frame)

 
