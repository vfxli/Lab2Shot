"""The farm: one queue for every user's cooks, run on the GPUs the administrator authorized.

clients   who started a job: its account, and what the request showed (address, application)
gpus      this machine's GPUs and which of them take jobs (the database record; farm/scheduler reads it)
scheduler the machine's places for nodes (each authorized GPU, the CPU slots) and which task's node gets one: live
          inventory, architecture fit, VRAM, queue order, limits, the time limit of a node
load      how busy the machine's CPU, memory and disk are (numbers the queue shows everyone)
queue     the tasks: their order (插队), cancelling, their records and the job log
timings   how long nodes take, and estimates from that
usage     how much each project and node is used, by department and by account, from the job log
disk      what task folders, the cache and uploads take, and cleaning what is unused
cards     the administrator's view of the cards, and what authorising other cards would change
policy    the administrator's settings the queue and the scheduler go by
tasks     background work that is not a node graph (installing an extension)
streaming a streaming node's worker on a thread of the farm

The records are in the database (lab2shot/database).
"""

from .clients import Client
from .queue import Farm, Job, farm, forget_job, history, job_row

__all__ = ["Client", "Farm", "Job", "farm", "forget_job", "history", "job_row"]
