"""The farm: one queue for every user's cooks, run on the GPUs the administrator authorized.

clients   who started a job: its account, and what the request showed (address, application)
gpus      this machine's GPUs and which of them take jobs (the database record; farm/scheduler reads it)
scheduler which GPU (if any) a job should run on: live inventory, architecture fit, VRAM, placement (place())
load      how busy the machine's CPU, memory and disk are (numbers the queue shows everyone)
queue     the jobs, in their lanes (light at once, heavy CPU a few at a time, one per authorized GPU), and the job log
timings   how long nodes take, and estimates from that
usage     how much each project and node is used, by department and by account, from the job log
disk      what the cache, uploads and deliveries take, and cleaning what is unused

The records are in the database (lab2shot/database).
"""

from .clients import Client
from .queue import Farm, Job, farm, forget_job, history, job_row

__all__ = ["Client", "Farm", "Job", "farm", "forget_job", "history", "job_row"]
