import csv
import subprocess
import os
from tqdm import tqdm
import pandas as pd
import urllib.request
from func_timeout import func_set_timeout
import time
import datetime
import func_timeout
# import config
picture = ["jpg","JPEG","PNG","png"]

def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s: %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 reset=True,
                                 log_colors={
                                     'DEBUG': 'blue',
                                     'INFO': 'green',
                                     'WARNING': 'yellow',
                                     'ERROR': 'red',
                                     'CRITICAL': 'red',
                                 })

    logger = logging.getLogger('project')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger

class open_image_dataset:
    def __init__(self):
        self.test_annotations_human = '/data/glusterfs_cv_04/public_data/imagenet/OpenImage/Human-verified_labels/test-annotations-human-imagelabels.csv'
        self.validation_annotations = '/data/glusterfs_cv_04/public_data/imagenet/OpenImage/Human-verified_labels/validation-annotations-human-imagelabels.csv'
        self.train_annotations = "/data/glusterfs_cv_04/public_data/imagenet/OpenImage/Human-verified_labels/oidv6-train-annotations-human-imagelabels.csv"

        self.Trainable = '/data/glusterfs_cv_04/public_data/imagenet/OpenImage/9600.csv'

        self.label_to_path = '/data/glusterfs_cv_04/public_data/imagenet/OpenImage/Image_IDs/oidv6-train-images-with-labels-with-rotation.csv'
        self.train_label_to_path_list = pd.read_csv(self.label_to_path)

        self.label_to_path_val = '/data/glusterfs_cv_04/public_data/imagenet/OpenImage/Image_IDs/validation-images-with-rotation.csv'
        self.val_label_to_path_list = pd.read_csv(self.label_to_path_val)

        self.path = "/data/glusterfs_cv_04/public_data/imagenet/OpenImage"

        self.class_descriptions = "/data/glusterfs_cv_04/public_data/imagenet/OpenImage/oidv6-class-descriptions.csv"
        self.class_descriptions_list = pd.read_csv(self.class_descriptions)

    def find_right_class(self):
        test_image_label = pd.read_csv(self.test_annotations_human)
        test_class_list = test_image_label["LabelName"].unique()

        val_image_label = pd.read_csv(self.validation_annotations)
        val_class_list = val_image_label["LabelName"].unique()

        train_image_label = pd.read_csv(self.train_annotations)
        train_class_list = train_image_label["LabelName"].unique()

        Trainable_label = pd.read_csv(self.Trainable)
        Trainable_class_list = Trainable_label["/m/01g317"].unique()

        final_class_list = list(
            set(test_class_list).intersection(val_class_list, train_class_list, Trainable_class_list))
        return final_class_list, train_image_label, val_image_label, test_image_label

    @func_set_timeout(50)
    def get_train_url(self,one_image_id,class_path):
        one_path = list(self.train_label_to_path_list[self.train_label_to_path_list["ImageID"] == one_image_id][
                            "Thumbnail300KURL"])[0]

        try:
            file_suffix = one_path.split('/')[-1]
            if file_suffix.split(".")[-1] not in picture:
                logger.info("invalid image url:"+one_path)
                return 0
        except:
            logger.info("invalid image url:"+str(one_path))
            return 0


        filename = class_path + "/" + file_suffix
        try:
            urllib.request.urlretrieve(one_path, filename=filename)
        except:
            logger.info("invalid image url:"+one_path)
            return 0
        return 0

    @func_set_timeout(50)
    def get_val_url(self, one_image_id, class_path):
        one_path = list(self.val_label_to_path_list[self.val_label_to_path_list["ImageID"] == one_image_id][
                            "Thumbnail300KURL"])[0]

        try:
            file_suffix = one_path.split('/')[-1]
            if file_suffix.split(".")[-1] not in picture:
                logger.info("invalid image url:" + one_path)
                return 0
        except:
            logger.info("invalid image url:" + str(one_path))
            return 0

        filename = class_path + "/" + file_suffix
        try:
            urllib.request.urlretrieve(one_path, filename=filename)
        except:
            logger.info("invalid image url:" + one_path)
            return 0
        return 0

    def download_url(self,class_path,one_class_list):

        for i,(one_image_id) in tqdm(enumerate(one_class_list)):
            logger.info("                download picture"+one_image_id)
            try:
                self.get_train_url(one_image_id,class_path)
            except func_timeout.exceptions.FunctionTimedOut:
                logger.info('Timed out!')
                continue

            if i > 600:
                break

    def download_url_val(self,class_path,one_class_list):

        for i,(one_image_id) in tqdm(enumerate(one_class_list)):
            logger.info("                download picture"+one_image_id)
            try:
                self.get_val_url(one_image_id,class_path)
            except func_timeout.exceptions.FunctionTimedOut:
                logger.info('Timed out!')
                continue

            if i > 50:
                break

    def download_train_image(self):
        class_list, train_image_label, val_image_label, test_image_label = self.find_right_class()
        logger.info("-------------------start download train data------------------")
        logger.info("                     class number: {:.1f}".format(len(class_list)))
        for class_one in class_list:
            DisplayName = \
            list(self.class_descriptions_list[self.class_descriptions_list["LabelName"] == class_one]["DisplayName"])[0]
            logger.info("-------------------start download new class------------------")
            logger.info("             LabelName: "+class_one+"     DisplayName:"+DisplayName)

            same_class = train_image_label[train_image_label["LabelName"] == class_one]
            confidence = same_class[same_class["Source"] == "verification"]
            one_class_list_clean = list(confidence[confidence["Confidence"] == 1]["ImageID"])
            one_class_list_noise = list(confidence[confidence["Confidence"] == 0]["ImageID"])
            if len(one_class_list_clean)>2000:
                logger.info("        warning:      invalid class")
                continue

            logger.info("                    all sample number : {:.1f}".format(len(confidence["ImageID"])))
            logger.info("       Clean sample number : {:.1f}".format(len(one_class_list_clean))+"  Noise sample number: {:.1f}".format(len(one_class_list_noise)))

            # one_class_list = list(train_image_label[train_image_label["LabelName"] == class_one]["ImageID"])
            logger.info("-------------------start download clean dataset------------------")
            class_path = os.path.join(self.path, "train","clean", class_one.split("/")[-1])
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            else:
                continue
            self.download_url(class_path,one_class_list_clean)

            logger.info("-------------------start download noise dataset------------------")
            class_path = os.path.join(self.path, "train","noise", class_one.split("/")[-1])
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            else:
                continue
            self.download_url(class_path, one_class_list_noise)

    def download_val_image(self):
        class_list, train_image_label, val_image_label, test_image_label = self.find_right_class()
        logger.info("-------------------start download val data------------------")
        logger.info("                     class number: {:.1f}".format(len(class_list)))

        class_path = os.path.join(self.path, "train", "clean")
        train_class_list = os.listdir(class_path)
        for class_one in class_list:

            if class_one.split("/")[-1] not  in train_class_list:
                continue

            DisplayName = \
            list(self.class_descriptions_list[self.class_descriptions_list["LabelName"] == class_one]["DisplayName"])[0]
            logger.info("-------------------start download new class------------------")
            logger.info("             LabelName: "+class_one+"     DisplayName:"+DisplayName)

            same_class = val_image_label[val_image_label["LabelName"] == class_one]
            confidence = same_class[same_class["Source"] == "verification"]
            one_class_list_clean = list(confidence[confidence["Confidence"] == 1]["ImageID"])

            logger.info("      all sample number : {:.1f}".format(len(confidence["ImageID"])))
            logger.info("       Clean sample number : {:.1f}".format(len(one_class_list_clean)))

            logger.info("-------------------start download dataset------------------")
            class_path = os.path.join(self.path, "val", class_one.split("/")[-1])
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            else:
                continue
            self.download_url_val(class_path,one_class_list_clean)



logger = setup_logger(os.path.join("/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_open_image/datasets", 'train_val_log'))
dataset=open_image_dataset()
dataset.download_val_image()

