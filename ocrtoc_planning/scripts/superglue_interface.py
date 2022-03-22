import sys
sys.path.append('../..')
from tqdm import tqdm
import rospkg
import numpy as np
import sys
import trimesh
import time
import json
import os
from include.io_interface import load_pickle, get_view_name, dump_pickle
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import read_image, convert_image
from utils import get_3d_points, get_projection_image, flann_matching, quaternion_matrix, load_object_list, render_image
PATH, OUTPATH = sys.argv[1], sys.argv[2]

def get_match(image, target_object_list):
    seg_image = image #segmented_image
    obj_list = target_object_list.keys() #list of objects
    list = []
    #get the match part not the pose
    for o in obj_list:
        o = o.decode("utf-8") 
        list.append(o)
    print(list)
    obj_list = list
    config = {
        'superpoint': {'nms_radius': 4, 'keypoints_threshold': 0.005, 'max_keypoints': 1024},
        'superglue': {'weights': 'outdoor',  'sinkhorn_iterations': 20, 'match_threshold': 0.2}
    }
    
    matching = Matching(config).eval().cuda()
    resize = [640, 360]
    device = 'cuda'
    rot = 0
    rendered_object_dir = os.path.join(
        rospkg.RosPack().get_path('ocrtoc_perception'),
        'src/ocrtoc_perception/pose/rendered_object_images'
    )
    models_dir = os.path.join(
    rospkg.RosPack().get_path('ocrtoc_materials'),
    'models'    ) 
    realsense_radius = 0.4
    camera_info = {
        'D': [0.0, 0.0, 0.0, 0.0, 0.0],
        'K': [917.9434734500945, 0.0, 639.5, 0.0, 917.9434734500945, 359.5, 0.0, 0.0, 1.0],
        'image_height': 720,
        'image_width': 1280
                }
    rendered_image_suffix = '{}_{}'.format(camera_info['image_height'], camera_info['image_width'])
    renderer = render_image(camera_info, models_dir, obj_list,
                        rendered_object_dir, rendered_image_suffix, realsense_radius)
    
    object_max = None
    match_max = 0
    image, inp, scale = convert_image(seg_image, device, resize, rot, False)
    for obj in tqdm(obj_list, 'calculating the match'):
        
        #keypoints from the seg image
        
        kps1_dict = matching.extract_feat(inp)
        
        kp1 = kps1_dict['keypoints'][0].cpu().numpy()
        if rot == 2:
            kp1[:, 0] = image.shape[1]-1-kp1[:, 0]
            kp1[:, 1] = image.shape[0]-1-kp1[:, 1]
        kp1 *= scale
        
        print(obj)
        
        # object_name = obj.decode("utf-8")
        object_name = obj.rsplit('_', 1)[0]
        
        template_id = 1
        template_name = get_view_name(template_id)
        base_dir = os.path.join(rendered_object_dir, object_name+rendered_image_suffix)
        cache_feature_path = os.path.join(
                base_dir, "{}_features.pickle".format(template_name))
        
        t_image, t_inp, t_scale = read_image(os.path.join(
                    base_dir, "{}.png".format(template_name)), 'cuda', resize, 0, False)

        print(cache_feature_path)
        
        raw_mesh_path = os.path.join(models_dir, object_name, 'visual.ply')

        raw_mesh = trimesh.load(raw_mesh_path)

        for template_id in range(0, len(renderer.views), 2):
            template_name = get_view_name(template_id)

            base_dir = os.path.join(rendered_object_dir, object_name+rendered_image_suffix)

            cache_feature_path = os.path.join(
                base_dir, "{}_features.pickle".format(template_name))

            t_image, t_inp, t_scale = read_image(os.path.join(
                    base_dir, "{}.png".format(template_name)), 'cuda', resize, 0, False)

            if os.path.exists(cache_feature_path):
                kps2_dict = load_pickle(cache_feature_path)
            else:
                kps2_dict = matching.extract_feat(t_inp)

                dump_pickle(kps2_dict, cache_feature_path)
        
        t1 = time.time()
        kp2 = kps2_dict['keypoints'][0].cpu().numpy()
        kp2 *= t_scale
            
        pred = {k+'0': v for k, v in kps1_dict.items()}
        pred['image0'] = inp
        pred = {**pred, **{k+'1': v for k, v in kps2_dict.items()}}
        pred['image1'] = t_inp
        pred = matching(pred)
        matches = pred['matches1'].cpu().numpy()
        print("printing matches", matches)
        
        match_len = np.nonzero(matches+1)[0].shape[0]
                
        print("Need to check match quality........")
        
        #compare max match and figure the object
        if match_max < match_len:
            object_max = obj
            match_max = match_len
            
        match_idx_kps1 = np.nonzero(matches+1)
        # print("match_idx_kps1", match_idx_kps1)
        # print("kps1", kp1)
        # kp1 = kp1[match_idx_kps1]
        # print("kps1[match_idx_kps1]", kp1)
        # match_idx_kps1 = matches[match_idx_kps1]
        # print('match_idx_kps1', match_idx_kps1)
        # kp2 = kp2[match_idx_kps1]
        # print('kps2', kp2)
        
        t2 = time.time()
        print('time elapsed in matching: {}'.format(t2-t1))
        print("current match len", match_len, "max match len", match_max)

    return object_max
    
if __name__ == "__main__":
    data = np.load(PATH, encoding='bytes', allow_pickle=True)['data'].item()
      
    image, target_object_list = data[b'image'], data[b'target_object_list']
    
    obj = [get_match(image, target_object_list)]
    np.savez_compressed(OUTPATH, data=obj)