"""
Re-link finished analyses that never got an ownership row.

An analysis writes its artefacts to processed/<job_id>/ and only then records
the owning user in the `analyses` table. If that INSERT fails — a dropped Neon
connection is the usual cause — the job is complete on disk but invisible:
/analyses omits it and /report/<job_id> answers 404. This re-creates the
missing rows from job_result.json and report.json.

    python relink_analyses.py                       # list orphaned jobs
    python relink_analyses.py --owner alice@demo.local --all
    python relink_analyses.py --owner 5 --job 115c7a34

Nothing is written without --all or --job. Jobs that already have a row are
never touched, so re-running is safe.
"""

import argparse
import datetime as dt
import json
import os
import sys

from auth import Analysis, SessionLocal, User, init_db

PROCESSED_DIR = "processed"
UPLOAD_DIR = "uploads"


def original_filename(job_id):
    """Recover the upload name from uploads/<job_id>_<filename>.

    job_result.json doesn't store it, but /analyze saves the upload under that
    prefix, so it's still there unless the file was cleaned up.
    """
    prefix = f"{job_id}_"
    try:
        names = os.listdir(UPLOAD_DIR)
    except OSError:
        return None

    for name in sorted(names):
        # Skip the converted WAV that sits alongside the original.
        if name.startswith(prefix) and not name.endswith("_processed.wav"):
            return name[len(prefix):]
    return None


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def find_jobs(db):
    """Every on-disk job, tagged with whether the DB already knows about it."""
    if not os.path.isdir(PROCESSED_DIR):
        return []

    owned = {job_id for (job_id,) in db.query(Analysis.job_id).all()}
    jobs = []

    for job_id in sorted(os.listdir(PROCESSED_DIR)):
        folder = os.path.join(PROCESSED_DIR, job_id)
        result = _load_json(os.path.join(folder, "job_result.json"))
        if result is None:
            continue  # not a finished job

        report = _load_json(os.path.join(folder, "report.json")) or {}
        jobs.append({
            "job_id": job_id,
            "linked": job_id in owned,
            "result": result,
            "filename": original_filename(job_id) or f"{job_id} (recovered)",
            "total_score": report.get("total_score"),
            "mtime": dt.datetime.fromtimestamp(os.path.getmtime(folder)),
        })

    return jobs


def resolve_owner(db, owner):
    """Accept either a user id or an email address."""
    q = db.query(User)
    user = q.filter(User.id == int(owner)).first() if owner.isdigit() else \
        q.filter(User.email == owner.strip().lower()).first()
    if user is None:
        sys.exit(f"No such user: {owner}. Run `python seed_users.py --list` to see accounts.")
    return user


def relink(db, job, user):
    result = job["result"]
    db.merge(Analysis(
        job_id=job["job_id"],
        owner_id=user.id,
        filename=job["filename"],
        created_at=job["mtime"],
        total_speakers=result.get("total_speakers"),
        total_segments=result.get("total_segments"),
        total_duration=result.get("total_duration"),
        lead_speaker=result.get("lead_speaker"),
        total_score=job["total_score"],
    ))
    db.commit()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", help="User id or email to assign the recovered jobs to.")
    ap.add_argument("--job", action="append", default=[],
                    help="Re-link this job id (repeatable).")
    ap.add_argument("--all", action="store_true", help="Re-link every orphaned job.")
    args = ap.parse_args()

    init_db()
    db = SessionLocal()
    try:
        jobs = find_jobs(db)
        orphans = [j for j in jobs if not j["linked"]]

        if not jobs:
            print("No finished jobs found under processed/.")
            return

        print(f"{'JOB':<12} {'STATUS':<10} {'SEGMENTS':>8} {'SCORE':>7}  FINISHED")
        for j in jobs:
            score = "-" if j["total_score"] is None else f"{j['total_score']:.1f}"
            print(f"{j['job_id']:<12} {'orphaned' if not j['linked'] else 'linked':<10} "
                  f"{j['result'].get('total_segments', '-'):>8} {score:>7}  "
                  f"{j['mtime']:%Y-%m-%d %H:%M}")

        if not orphans:
            print("\nEvery job on disk is linked to a user — nothing to do.")
            return

        if not (args.all or args.job):
            print(f"\n{len(orphans)} orphaned job(s). Re-link them with:")
            print("  python relink_analyses.py --owner <email-or-id> --all")
            return

        if not args.owner:
            sys.exit("--owner is required when re-linking.")

        user = resolve_owner(db, args.owner)
        targets = orphans if args.all else [j for j in orphans if j["job_id"] in args.job]

        unknown = set(args.job) - {j["job_id"] for j in orphans}
        for job_id in sorted(unknown):
            print(f"  ! {job_id} is not an orphaned job — skipped")

        print(f"\nRe-linking {len(targets)} job(s) to {user.email}:")
        for job in targets:
            relink(db, job, user)
            print(f"  + {job['job_id']}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
