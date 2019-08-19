from traffic_analysis.d00_utils.load_confs import load_paths
import subprocess

paths = load_paths()


def move_file(old_path, new_path):
    bucket_name = paths['bucket_name']
    s3_profile = paths['s3_profile']

    try:
        if old_path:
            old_filename = "s3://%s/%s" % (bucket_name, old_path)
            new_filename = "s3://%s/%s" % (bucket_name, new_path)
            res = subprocess.call(["aws", "s3", 'mv',
                                   old_filename,
                                   new_filename,
                                   '--profile',
                                   s3_profile])
    except Exception as e:
        print(e)
        return False
    return True


if __name__ == "__main__":
    move_file("raw/videos/2019-08-15/20190815-001725_00001.07555.mp4",
              "raw/processed_videos/2019-08-15/20190815-001725_00001.07555.mp4")
