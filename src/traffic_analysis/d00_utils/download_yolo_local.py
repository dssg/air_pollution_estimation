import os
import urllib

def download_yolo_local(local_dir):

    download_dict = {os.path.join(local_dir, "yolov3-tiny"): ["https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
                                                                  "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
                                                                  "https://pjreddie.com/media/files/yolov3-tiny.weights"],
                         os.path.join(local_dir, "yolov3"): ["https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
                                                             "https://pjreddie.com/media/files/yolov3.weights",
                                                             "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
                                                             "https://raw.githubusercontent.com/wizyoung/YOLOv3_TensorFlow/master/data/yolo_anchors.txt"
                                                             ]
                         }

    for download_dir, download_urls in download_dict.items():

        if(not os.path.isdir(download_dir)):
            os.makedirs(download_dir)
            for download_url in download_urls:
                filename = download_url.split("/")[-1]
                download_path = os.path.join(download_dir, filename)

                try:
                    urllib.request.urlretrieve(download_url,
                                               download_path)
                    print(f"Successfully downloaded {download_url}")

                except Exception as e:
                    print(e)
                    print("Failed to download url ", download_url)
        else:
            print('Model already downloaded!')

    return
