from crontab import CronTab
import os
from traffic_analysis.d00_utils.load_confs import load_paths
from pathlib import Path

home = Path.home()
paths = load_paths()
python_path = "python3"


class CronJobs:
    def __init__(self, username):
        sys_path = os.environ.get("PATH")
        self.cron = CronTab(user=username,
                            log=os.path.join("tmp", "cron.log"))
        self.cron.env['PATH'] = sys_path

    def pipeline_job(self):
        filename = home.joinpath(
            "air_pollution_estimation", "src", "live_pipeline.py")
        print(filename)
        job = self.cron.new(
            command=f"{python_path} {filename} >> /tmp/pipeline.log 2>&1", comment="pipeline")
        job.minute.every(4)
        self.cron.write_to_user()

    def download_videos_job(self):
        filename = home.joinpath(
            "air_pollution_estimation", "src", "download_videos.py")
        job = self.cron.new(
            command=f"{python_path} {filename} >> /tmp/download.log 2>&1", comment="download")
        job.minute.every(4)
        self.cron.write()

    def upload_videos_job(self):
        filename = home.joinpath(
            "air_pollution_estimation", "src", "upload_videos.py")
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

    def remove_all_jobs(self):
        self.cron.remove_all()
        self.cron.write()

    def remove_jobs_by_comment(self, job_comment):
        self.cron.remove_all(comment=job_comment)


if __name__ == "__main__":
    import getpass
    username = getpass.getuser()
    cron_jobs = CronJobs(username)
    cron_jobs.remove_jobs_by_comment('pipeline')
    # cron_jobs.download_videos_job()
    # cron_jobs.upload_videos_job()
    cron_jobs.pipeline_job()
    cron_jobs.view_jobs()
