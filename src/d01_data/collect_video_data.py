import urllib.request
import datetime
import os
import subprocess
import time
import smtplib
import ssl
import configparser


def download_jam_cams(website, camera, extension, video_dir):

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H.%M")
    url = website + camera + extension
    file_path = video_dir + "/" + camera + "/" + timestamp + extension
    urllib.request.urlretrieve(url, file_path)


def create_folder_structure(local_video_dir: str,
                            camera_list: list):
    if not os.path.exists(local_video_dir):
        os.makedirs(local_video_dir)
    for camera in camera_list:
        this_camera_dir = os.path.join(local_video_dir, camera)
        if not os.path.exists(this_camera_dir):
            os.makedirs(this_camera_dir)


def get_videos_and_upload_to_s3(local_video_dir: str,
                                camera_list: list,
                                website: str = "http://jamcams.tfl.gov.uk/00001.",
                                extension: str ='.mp4',
                                ):

    create_folder_structure(local_video_dir=local_video_dir,
                            camera_list=camera_list)

    for camera in camera_list:
        download_jam_cams(website, camera, extension, local_video_dir)

    time.sleep(1 * 60)
    res = subprocess.call(["aws", "s3", 'cp',
                           local_video_dir,
                           's3://air-pollution-uk/raw/video_data/',
                           '--recursive',
                           '--profile',
                           'air-quality'])
    res = subprocess.call(["rm", "-r",
                           local_video_dir
                           ])


def collect_video_data(local_video_dir: str,
                       camera_list: list,
                       num_iterations: int = None,
                       website: str = "http://jamcams.tfl.gov.uk/00001.",):

    upload_num = 0

    if num_iterations is None:
        print('Starting infinite data collection.')
        while True:
            try:
              get_videos_and_upload_to_s3(local_video_dir=local_video_dir,
                                          camera_list=camera_list,
                                          website=website)
              print('Completed Iteration')
              time.sleep(3 * 60)

            except:

              setup_dir = os.path.join(os.getcwd(), '..', '..')
              config = configparser.ConfigParser()
              config.read(os.path.join(setup_dir, 'conf', 'local', 'credentials.yml'))
              
              sender_email = config.get('EMAIL', 'address')
              password = config.get('EMAIL', 'password')
              recipients = config.get('EMAIL', 'recipients')

              port = 465  # For SSL
              smtp_server = "smtp.gmail.com"
              message = """\
              Subject: ERROR - Traffic Camera Download Failed

              The script responsible for downloading the traffic camera data has been stopped."""

              context = ssl.create_default_context()
              with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                  server.login(sender_email, password)
                  server.sendmail(sender_email, recipients, message)

              print('Download Failed!')
              break;

    else:
        print('Starting data collection.')
        while upload_num < num_iterations:
            get_videos_and_upload_to_s3(local_video_dir=local_video_dir,
                                        camera_list=camera_list,
                                        website=website)
            upload_num += 1
            print('Completed {}/{} iterations'.format(upload_num,
                                                      num_iterations))
            time.sleep(3 * 60)






