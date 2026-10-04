"""Deleting an account and deleting it permanently: what goes with it from every part of the site, in order.

The account itself (its row, sessions, what the database keeps of it) is lab2shot/accounts.py's; what else belongs to it
is the parts' above it: its jobs (farm), its outputs and uploads (transfer, farm/disk.py), its saved graphs (library.py)
and the feedback bundles naming it (feedback.py). This module is above all of them (cli/check_arch.py LAYERS: site), so
accounts.py never reaches up to them.

    delete(user_id)  switched off and its sessions end (nothing new comes from it), its live jobs stop and are waited
                     for until they have ended (nothing of theirs is written any more; E-QUEUE-STILLSTOPPING, the
                     account left switched off, when one does not end), then its tasks' outputs are removed (its cache,
                     uploads and task folders go by housekeeping: farm/disk.py) and it is marked deleted. Its job records
                     remain for the statistics (as 「已删除的用户」) and its feedback remains for the administrator.
    purge(user_id)   an already deleted account is removed for good (accounts.purge_rows says what is kept), with its
                     saved graphs, tasks, cache and uploads, and no feedback bundle names it any more."""

from __future__ import annotations

from .. import accounts
from ..errors import Invalid
from ..messages import Msg


def delete(user_id: int) -> dict:
    """Delete an account (never the built-in administrator account); returns what was removed."""
    from ..farm import farm
    from ..transfer import outputs

    accounts.switch_off_to_delete(user_id)
    stopped = farm().stop_user(user_id)
    gone = outputs.remove_account(user_id)
    accounts.mark_deleted(user_id)
    return {"jobs_stopped": stopped, "outputs": gone}


def purge(user_id: int) -> dict:
    """Permanently delete an already deleted account. Returns the number of rows kept per table and the graphs removed,
    which the confirmation dialog and the audit log report."""
    from ..farm import farm
    from ..farm.disk import forget_account
    from .feedback import forget_in_bundles
    from .library import remove_user

    u = accounts.purgeable(user_id)
    # a job of it still to finish would write into (and make again) the cache and folders removed below; a deleted
    # account gets no new one (no session, and a held job is not queued again for it: farm/queue.py _unpark)
    if live := farm().live_of(user_id):
        raise Invalid(Msg("E-ACCOUNT-JOBSLIVE", username=u.username, count=len(live)))
    kept = accounts.kept_counts(user_id)
    graphs = remove_user(u.username)
    forget_account(user_id)  # its tasks, cache and uploads
    forget_in_bundles(u)  # before purge_rows forgets where it logged in from (accounts.marked reads login_log)
    accounts.purge_rows(u)
    return {"jobs": kept["jobs"], "feedback": kept["feedback"], "graphs": graphs}
