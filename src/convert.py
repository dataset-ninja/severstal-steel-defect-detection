# https://github.com/zdaiot/Kaggle-Steel-Defect-Detection

import os
import shutil
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
    mkdir,
    remove_dir,
)
from tqdm import tqdm

import src.settings as s


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "severstal-steel-defect"
    dataset_path = "/mnt/d/datasetninja-raw/severstal"
    masks_path = "/home/alex/DATASETS/TODO/severstal-steel-defect/archive_masks"
    mask_suffix = "_mask.png"
    images_ext = ".jpg"
    batch_size = 30

    def create_ann(image_path):
        labels = []

        mask_name = get_file_name(image_path) + mask_suffix
        mask_path = os.path.join(masks_path, mask_name)

        mask_np = sly.imaging.image.read(mask_path)[:, :, 0]

        img_height = mask_np.shape[0]
        img_wight = mask_np.shape[1]

        unique_pixels = np.unique(mask_np)[1:]
        for curr_pixel in unique_pixels:
            obj_class = pixel_to_class[curr_pixel]
            mask = mask_np == curr_pixel
            ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
            for i in range(1, ret):
                obj_mask = curr_mask == i
                curr_bitmap = sly.Bitmap(obj_mask)
                if curr_bitmap.area > 100:
                    curr_label = sly.Label(curr_bitmap, obj_class)
                    labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    obj_class_1 = sly.ObjClass("defect_1", sly.Bitmap)
    obj_class_2 = sly.ObjClass("defect_2", sly.Bitmap)
    obj_class_3 = sly.ObjClass("defect_3", sly.Bitmap)
    obj_class_4 = sly.ObjClass("defect_4", sly.Bitmap)

    pixel_to_class = {1: obj_class_1, 2: obj_class_2, 3: obj_class_3, 4: obj_class_4}

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class_1, obj_class_2, obj_class_3, obj_class_4])
    api.project.update_meta(project.id, meta.to_json())

    for ds_name in os.listdir(dataset_path):
        curr_ds_path = os.path.join(dataset_path, ds_name)

        if dir_exists(curr_ds_path):
            dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

            images_names = os.listdir(curr_ds_path)

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

            for img_names_batch in sly.batched(images_names, batch_size=batch_size):
                img_pathes_batch = [
                    os.path.join(curr_ds_path, im_name) for im_name in img_names_batch
                ]

                img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                if ds_name == "train_images":
                    anns = [create_ann(image_path) for image_path in img_pathes_batch]
                    api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(img_names_batch))
    return project
