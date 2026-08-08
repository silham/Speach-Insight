"""
Seed login accounts. There is no signup endpoint — this is the only way
users are created.

    python seed_users.py                  # create the default demo accounts
    python seed_users.py --list           # show existing accounts
    python seed_users.py --email a@b.c --name "Ada" --role admin [--password pw]
    python seed_users.py --reset-password --email a@b.c --password newpw

Passwords are printed once on creation. Existing accounts are left alone
unless you pass --reset-password.
"""

import argparse
import secrets
import sys

from auth import ROLES, SessionLocal, User, hash_password, init_db

DEFAULT_USERS = [
    {"email": "admin@demo.local", "name": "Admin", "role": "admin"},
    {"email": "alice@demo.local", "name": "Alice", "role": "user"},
    {"email": "bob@demo.local", "name": "Bob", "role": "user"},
]


def upsert(db, email, name, role, password=None, reset=False):
    email = email.strip().lower()
    existing = db.query(User).filter(User.email == email).first()

    if existing and not reset:
        print(f"  = {email:<22} already exists (role={existing.role}) — skipped")
        return None

    plain = password or secrets.token_urlsafe(9)

    if existing:
        existing.password_hash = hash_password(plain)
        db.commit()
        print(f"  ~ {email:<22} password reset            -> {plain}")
        return plain

    db.add(User(email=email, name=name, role=role, password_hash=hash_password(plain)))
    db.commit()
    print(f"  + {email:<22} created (role={role})".ljust(52) + f"-> {plain}")
    return plain


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--email")
    ap.add_argument("--name")
    ap.add_argument("--role", choices=ROLES, default="user")
    ap.add_argument("--password", help="If omitted, a random one is generated and printed.")
    ap.add_argument("--reset-password", action="store_true", help="Overwrite the password of an existing account.")
    ap.add_argument("--list", action="store_true", help="List accounts and exit.")
    args = ap.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if args.list:
            users = db.query(User).order_by(User.id).all()
            if not users:
                print("No users yet. Run `python seed_users.py` to create the demo accounts.")
                return
            print(f"{'ID':<4} {'EMAIL':<26} {'NAME':<16} ROLE")
            for u in users:
                print(f"{u.id:<4} {u.email:<26} {u.name:<16} {u.role}")
            return

        if args.reset_password and not args.email:
            sys.exit("--reset-password requires --email")

        if args.email:
            print("Seeding user:")
            upsert(
                db,
                args.email,
                args.name or args.email.split("@")[0].title(),
                args.role,
                args.password,
                reset=args.reset_password,
            )
        else:
            print("Seeding default demo accounts:")
            for spec in DEFAULT_USERS:
                upsert(db, **spec)

        print("\nSave these passwords — they are hashed and cannot be recovered.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
