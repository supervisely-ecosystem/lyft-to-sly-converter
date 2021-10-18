try:
    from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
except ImportError:
    raise ImportError('Please run "pip install lyft_dataset_sdk" '
                      'to install the official devkit first.')

import os
import shutil

import numpy as np
import open3d as o3d
import supervisely_lib as sly
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from supervisely_lib.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
from supervisely_lib.project.pointcloud_project import OpenMode
from tqdm import tqdm


def _convert_label_to_geometry(label):
    geometries = []
    for l in label:
        bbox = l.to_xyzwhlr()
        dim = bbox[[3, 5, 4]]
        pos = bbox[:3] + [0, 0, dim[1] / 2]
        yaw = bbox[-1]
        position = Vector3d(float(pos[0]), float(pos[1]), float(pos[2]))
        rotation = Vector3d(0, 0, float(-yaw))

        dimension = Vector3d(float(dim[0]), float(dim[2]), float(dim[1]))
        geometry = Cuboid3d(position, rotation, dimension)
        geometries.append(geometry)
    return geometries


def convert_label_to_annotation(label, meta):
    geometries = _convert_label_to_geometry(label)
    figures = []
    objs = []
    for l, geometry in zip(label, geometries):  # by object in point cloud
        pcobj = sly.PointcloudObject(meta.get_obj_class(l.label_class))
        figures.append(sly.PointcloudFigure(pcobj, geometry))
        objs.append(pcobj)

    annotation = sly.PointcloudAnnotation(PointcloudObjectCollection(objs), figures)
    return annotation


def convert_bin_to_pcd(bin_file, save_filepath):
    bin = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
    points = bin[:, 0:3]
    intensity = bin[:, 3]
    ring_index = bin[:, 4]
    intensity_fake_rgb = np.zeros((intensity.shape[0], 3))
    intensity_fake_rgb[:, 0] = intensity  # red The intensity measures the reflectivity of the objects
    intensity_fake_rgb[:, 1] = ring_index  # green ring index is the index of the laser ranging from 0 to 31

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(intensity_fake_rgb)
    o3d.io.write_point_cloud(save_filepath, pc)


def lyft_annotation_to_BEVBox3D(data):
    boxes = data['gt_boxes']
    names = data['names']

    objects = []
    for name, box in zip(names, boxes):
        center = [float(box[0]), float(box[1]), float(box[2])]
        size = [float(box[3]), float(box[5]), float(box[4])]
        ry = float(box[6])

        yaw = ry - np.pi
        yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi
        world_cam = None
        objects.append(o3d.ml.datasets.utils.BEVBox3D(center, size, yaw, name, -1.0, world_cam))
        objects[-1].yaw = ry

    return objects


def get_available_scenes(lyft):
    available_scenes = []

    for scene in lyft.scene:
        token = scene['token']
        scene_rec = lyft.get('scene', token)
        sample_rec = lyft.get('sample', scene_rec['first_sample_token'])
        sample_data = lyft.get('sample_data',
                               sample_rec['data']['LIDAR_TOP'])

        lidar_path, boxes, _ = lyft.get_sample_data(sample_data['token'])
        if not os.path.exists(str(lidar_path)):
            continue
        else:
            available_scenes.append(scene)
    for i, a in enumerate(available_scenes):
        sly.logger.info(f'Scene: {i} Samples:{a["nbr_samples"]} Name: {a["name"]}')
    return available_scenes


def process_lyft_scene(lyft, scene):
    new_token = scene["first_sample_token"]
    my_sample = lyft.get('sample', new_token)

    dataset_data = []
    for i in tqdm(range(scene["nbr_samples"] - 1)):  # because pass first sample
        new_token = my_sample['next']  # TODO: not skip first frame
        my_sample = lyft.get('sample', new_token)

        data = {}
        data['timestamp'] = my_sample['timestamp']

        sensor_token = my_sample['data']['LIDAR_TOP']
        lidar_path, boxes, _ = lyft.get_sample_data(sensor_token)

        sd_record_lid = lyft.get("sample_data", sensor_token)
        cs_record_lid = lyft.get("calibrated_sensor", sd_record_lid["calibrated_sensor_token"])
        ego_record_lid = lyft.get("ego_pose", sd_record_lid["ego_pose_token"])

        assert os.path.exists(lidar_path)

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes
                         ]).reshape(-1, 1)

        names = np.array([b.name for b in boxes])
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2],
                                  axis=1)
        data['names'] = names
        data['lidar_path'] = str(lidar_path)
        data['gt_boxes'] = gt_boxes

        for sensor, sensor_token in my_sample['data'].items():
            if 'CAM' in sensor:
                img_path, boxes, cam_intrinsic = lyft.get_sample_data(sensor_token)
                assert os.path.exists(img_path)
                data[sensor] = str(img_path)

                sd_record_cam = lyft.get("sample_data", sensor_token)
                cs_record_cam = lyft.get("calibrated_sensor", sd_record_cam["calibrated_sensor_token"])
                ego_record_cam = lyft.get("ego_pose", sd_record_cam["ego_pose_token"])
                cam_height = sd_record_cam["height"]
                cam_width = sd_record_cam["width"]

                lid_to_ego = transform_matrix(
                    cs_record_lid["translation"], Quaternion(cs_record_lid["rotation"]), inverse=False
                )
                lid_ego_to_world = transform_matrix(
                    ego_record_lid["translation"], Quaternion(ego_record_lid["rotation"]), inverse=False
                )
                world_to_cam_ego = transform_matrix(
                    ego_record_cam["translation"], Quaternion(ego_record_cam["rotation"]), inverse=True
                )
                ego_to_cam = transform_matrix(
                    cs_record_cam["translation"], Quaternion(cs_record_cam["rotation"]), inverse=True
                )

                velo_to_cam = np.dot(ego_to_cam, np.dot(world_to_cam_ego, np.dot(lid_ego_to_world, lid_to_ego)))
                velo_to_cam_rot = velo_to_cam[:3, :3]
                velo_to_cam_trans = velo_to_cam[:3, 3]

                data[f'{sensor}_extrinsic'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
                data[f'{sensor}_intrinsic'] = np.asarray(cs_record_cam['camera_intrinsic'])
                data[f'{sensor}_imsize'] = (cam_width, cam_height)
            else:
                sly.logger.debug(f"pass {sensor} - isn't a camera")

        dataset_data.append(data)
    return dataset_data


class Superviselyft:
    def __init__(self, lyft, sly_project_path):
        self.lyft = lyft
        self.scenes = get_available_scenes(self.lyft)
        sly.logger.info(f"Lyft -> Sly. Available {len(self.scenes)} scenes.")

        shutil.rmtree(sly_project_path, ignore_errors=True)  # WARN!
        self.project_fs = sly.PointcloudProject(sly_project_path, OpenMode.CREATE)
        self.project_fs.set_meta(self.construct_cuboids_meta([x['name'] for x in lyft.category]))

    def construct_cuboids_meta(self, labels):
        geometry = Cuboid3d
        unique_labels = np.unique(labels)
        obj_classes = [sly.ObjClass(k, geometry) for k in unique_labels]
        meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        return meta

    def convert_scene_id(self, idx):
        dataset_data = process_lyft_scene(self.lyft, self.scenes[idx])
        dataset_fs = self.project_fs.create_dataset(f"Sequence_{idx}_{len(dataset_data)}")
        sly.logger.info(f"Created Supervisely dataset with {dataset_fs.name} at {dataset_fs.directory}")

        for data in dataset_data:
            item_name = sly.fs.get_file_name(data['lidar_path']) + ".pcd"
            item_path = dataset_fs.generate_item_path(item_name)

            label = lyft_annotation_to_BEVBox3D(data)
            ann = convert_label_to_annotation(label, self.project_fs.meta)

            convert_bin_to_pcd(data['lidar_path'], item_path)  # automatically save pointcloud to itempath

            dataset_fs.add_item_file(item_name, item_path, ann)
            sly.logger.info(f".bin -> {item_name}")

            related_images_path = dataset_fs.get_related_images_path(item_name)
            os.makedirs(related_images_path, exist_ok=True)

            for sensor, image_path in data.items():
                if 'CAM' in sensor and not sensor.endswith('_intrinsic') \
                        and not sensor.endswith("_extrinsic") \
                        and not sensor.endswith("_imsize"):
                    image_name = sly.fs.get_file_name_with_ext(image_path)
                    sly_path_img = os.path.join(related_images_path, image_name)
                    shutil.copy(src=image_path, dst=sly_path_img)
                    img_info = {
                        "name": image_name,
                        "meta": {
                            "deviceId ": sensor,
                            "timestamp": data['timestamp'],
                            "sensorsData": {
                                "extrinsicMatrix": list(data[f'{sensor}_extrinsic'].flatten().astype(float)),
                                "intrinsicMatrix": list(data[f'{sensor}_intrinsic'].flatten().astype(float))
                            }
                        }
                    }
                    sly.json.dump_json_file(img_info, sly_path_img + '.json')


if __name__ == "__main__":
    lyft = Lyft(data_path="/data/SAMPLE", json_path="/data/SAMPLE/data")

    slyft = Superviselyft(lyft, "/data/LyftPhotoContext")
    slyft.convert_scene_id(0)
