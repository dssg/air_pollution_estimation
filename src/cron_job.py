from crontab import CronTab
from pathlib import Path
import os
import sys
from traffic_analysis.d00_utils.load_confs import load_paths

paths = load_paths()
home = Path.home()
python_path = "python3"


class CronJobs:
    def __init__(self, username):
        # filename = "/tmp/crontabs/crontab.cron"
        # if not os.path.exists(filename):
        #     print(os.path.dirname(filename))
        #     os.makedirs(os.path.dirname(filename), exist_ok=True)
        sys_path = os.environ.get("PATH")
        print(sys_path)
        # with open(filename, "w+") as f:
        #     f.write(f"\n\nPATH={sys_path}\n\n")

        self.cron = CronTab(user=username,
                            log="/tmp/dssg_cron.log")
        self.cron.env['PATH'] = sys_path

    def pipeline_job(self):
        filename = home.joinpath(
            "air_pollution_estimation/src/live_pipeline.py")
        job = self.cron.new(
            command=f"{python_path} {filename} >> /tmp/pipeline.log 2>&1", comment="pipeline")
        job.minute.every(4)
        self.cron.write_to_user()

    def download_videos_job(self):
        filename = home.joinpath(
            "air_pollution_estimation/src/download_videos.py")
        job = self.cron.new(
            command=f"{python_path} {filename} >> /tmp/download.log 2>&1", comment="download")
        job.minute.every(4)
        self.cron.write()

    def upload_videos_job(self):
        filename = home.joinpath(
            "air_pollution_estimation/src/upload_videos.py")
        job = self.cron.new(
            command=f"{python_path} {filename} >> /tmp/upload.log 2>&1", comment="upload")
        job.minute.every(1)
        self.cron.write()

    def view_jobs(self):
        for job in self.cron:
            print(job)

    def view_log(self):
        for d in self.cron.log:
            print(d['pid'] + " - " + d['date'])

    def clear_all_jobs(self):
        self.cron.remove_all()
        self.cron.write()

    def test(self):
        filename = home.joinpath("air_pollution_estimation/src/test.py")
        filename = "python3 --version >> /tmp/cron.log"
        job = self.cron.new(
            command=f"{filename}", comment="test")
        job.minute.every(1)
        self.cron.write()


if __name__ == "__main__":
    import getpass
    username = getpass.getuser()
    cron_jobs = CronJobs(username)
    print(username)
    cron_jobs.clear_all_jobs()
    cron_jobs.download_videos_job()
    cron_jobs.test()
    cron_jobs.upload_videos_job()
    cron_jobs.pipeline_job()
    cron_jobs.view_jobs()
    cron_jobs.view_log()
