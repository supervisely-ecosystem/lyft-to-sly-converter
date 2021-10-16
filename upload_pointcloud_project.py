import os

import supervisely_lib as sly
import tqdm
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.io.json import load_json_file
from supervisely_lib.video_annotation.key_id_map import KeyIdMap

if __name__ == "__main__":
    api = sly.Api.from_env()
    base_dir = '/data/'
    project_name = "LyftPhotoContext"
    project_id = None
    max_n = 1000
    if project_id is not None:
        project = api.project.get_info_by_id(project_id)  # to existing project
    else:
        project = api.project.create(os.environ["context.workspaceId"],
                                     project_name,
                                     type=sly.ProjectType.POINT_CLOUDS,
                                     change_name_if_conflict=True)

    project_fs = sly.PointcloudProject.read_single(base_dir + project_name)

    api.project.update_meta(project.id, project_fs.meta.to_json())
    sly.logger.info("Project {!r} [id={!r}] has been created".format(project.name, project.id))

    uploaded_objects = KeyIdMap()

    for dataset_fs in project_fs:
        dataset = api.dataset.create(project.id, dataset_fs.name, change_name_if_conflict=True)
        sly.logger.info("dataset {!r} [id={!r}] has been created".format(dataset.name, dataset.id))
        i = 0
        for item_name in tqdm.tqdm(dataset_fs):

            if i >= max_n:
                break
            i += 1

            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)
            related_items = dataset_fs.get_related_images(item_name)

            try:
                _, meta = related_items[0]
                timestamp = meta[ApiField.META]['timestamp']
                if timestamp:
                    item_meta = {"timestamp": timestamp}
            except (KeyError, IndexError):
                item_meta = {}

            pointcloud = api.pointcloud.upload_path(dataset.id, item_name, item_path, item_meta)

            # validate_item_annotation
            ann_json = sly.io.json.load_json_file(ann_path)
            ann = sly.PointcloudAnnotation.from_json(ann_json, project_fs.meta)

            # ignore existing key_id_map because the new objects will be created
            api.pointcloud.annotation.append(pointcloud.id, ann, uploaded_objects)

            # upload related_images if exist
            if len(related_items) != 0:
                rimg_infos = []
                for img_path, meta_json in related_items:
                    img_name = sly.fs.get_file_name(img_path)
                    img = api.pointcloud.upload_related_image(img_path)[0]
                    rimg_infos.append({ApiField.ENTITY_ID: pointcloud.id,
                                       ApiField.NAME: meta_json[ApiField.NAME],
                                       ApiField.HASH: img,
                                       ApiField.META: meta_json[ApiField.META]})

                api.pointcloud.add_related_images(rimg_infos)

    sly.logger.info('PROJECT_UPLOADED', extra={'event_type': sly.EventType.PROJECT_CREATED, 'project_id': project.id})
