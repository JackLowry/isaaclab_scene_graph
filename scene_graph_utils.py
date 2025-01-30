import json
from tqdm import tqdm
import numpy as np
import pickle
import torch

edge_name_to_id = {
    "no_relation": 0,
    "right_of": 1,
    "left_of": 2,
    "on_top_of": 3,
    "below": 4,
    "in_front_of": 5,
    "behind": 6
}
edge_id_to_name = {edge_name_to_id[name]:name for name in edge_name_to_id}

edge_unit_vector_ordered = [
    [0, 0, 0],

    [0, -1, 0],  #right_of
    [0,  1, 0], #left_of

    [0, 0,  1],  #on_top_of
    [0, 0, -1], #below

    [-1,  0, 0],  #in_front_of
    [1, 0, 0]  #behind
]
edge_unit_vector_ordered = torch.Tensor(edge_unit_vector_ordered).cuda()


def get_bounding_box(mask):
    indices = mask.nonzero()
    if indices.shape[0] == 0:
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
    else:
        x_min = int(np.min(indices[1]))
        x_max = int(np.max(indices[1]))
        y_min = int(np.min(indices[0]))
        y_max = int(np.max(indices[0]))

    bbox = torch.Tensor((x_min, y_min, x_max, y_max))
    
    return bbox

def get_union_bounding_box(bbox1, bbox2):
    union_bbox = (min(bbox1[0], bbox2[0]), 
                  min(bbox1[1], bbox2[1]), 
                  max(bbox1[2], bbox2[2]), 
                  max(bbox1[3], bbox2[3]))
    
    return torch.Tensor(union_bbox)


def create_scene_graph(reset_obs, distance_cutoff = 0.2):

    graphs = []
    for graph_idx, obs in enumerate(reset_obs):
        rgb = obs["policy"]["images"]["rgb"]
        mask = obs["policy"]["images"]["mask"]
        depth = obs["policy"]["images"]["depth"]
        mask_metadata = obs["policy"]["extra_info"]["mask_info"]
        obj_metadata = obs["policy"]["extra_info"]["object_metadata"]
        obj_states = obs["policy"]["object_states"]

        mask_labels_to_id = {v:k for k,v in mask_metadata['idToLabels'].items()}


        nodes = {}
        graphs_env = []
        for env_id in range(obj_states.shape[0]):
            for obj_sim_name in obj_metadata.keys():
                obj_data = obj_metadata[obj_sim_name]
                sim_idx = obj_data["sim_idx"]
                if obj_sim_name in mask_labels_to_id.keys():
                    color_id = mask_labels_to_id[obj_sim_name]
                else: 
                    color_id = -1
                obj_mask = mask[env_id] == color_id
                img_bbox = get_bounding_box(obj_mask)



                obj_center = obj_states[env_id, sim_idx]

                # object is not placed yet
                if abs(obj_center[1]) >  0.8:
                    nodes[obj_sim_name] = {
                        "xyz_center": torch.full_like(obj_center, -1),
                        "class_name": "None",
                        "bbox": img_bbox,
                        "mask_color_id": -1,
                        # "object_id": -1
                    }
                else:
                    nodes[obj_sim_name] = {
                        "xyz_center": obj_center,
                        "class_name": obj_data["class"],
                        "bbox": img_bbox,
                        "mask_color_id": color_id,
                        # "object_class": mask_metadata["idToSemantics"][color_id]
                    }

            edges = {}

            for obj_sim_name in nodes.keys():
                for related_obj_sim_name in nodes.keys():
                    if obj_sim_name == related_obj_sim_name:
                        continue

                    if nodes[obj_sim_name]["class_name"] is None or nodes[related_obj_sim_name]["class_name"] is None:
                        edges[(obj_sim_name, related_obj_sim_name)] = {
                            "name": "no_relation",
                            "bbox": bbox,
                            "xyz_offset": torch.zeros_like(obj_center),
                            "relation_id": edge_name_to_id["no_relation"]
                        }

                    obj_data = obj_metadata[obj_sim_name]
                    sim_idx = obj_data["sim_idx"]
                    related_obj_data = obj_metadata[related_obj_sim_name]
                    related_sim_idx = related_obj_data["sim_idx"]
                    
                    relation = obj_states[env_id, sim_idx] - obj_states[env_id, related_sim_idx]

                    dist = torch.norm(relation)

                    if dist > distance_cutoff:
                        relationship_id = 0 #no relation
                    else: 
                        relation = relation[np.newaxis].T

                        relationship_components = edge_unit_vector_ordered@relation

                        relationship_id = torch.argmax(relationship_components).cpu().item()

                    bbox = get_union_bounding_box(nodes[obj_sim_name]["bbox"], nodes[related_obj_sim_name]["bbox"])

                    edges[(obj_sim_name, related_obj_sim_name)] = {
                        "name": edge_id_to_name[relationship_id],
                        "bbox": bbox,
                        "xyz_offset": relation,
                        "relation_id": relationship_id
                    }
            
            graph = {
                "nodes": nodes,
                "edges": edges,
                "imgs": {
                    "rgb": rgb,
                    "depth": depth,
                    "mask": mask
                }
            }
            graphs_env.append(graph)
        graphs.append(graphs_env)
    return graphs
    