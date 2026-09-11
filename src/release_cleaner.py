import os
import sys
import argparse
from datetime import datetime, timezone
import requests
from typing import Optional, List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import GITHUB_TOKEN, GITHUB_REPOSITORY

def clean_old_releases(
    days: int = 5,
    repo: Optional[str] = None,
    token: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Deletes GitHub Releases and assets that are older than `days` (default: 5 days).
    Prevents storage hoarding and keeps GitHub repository 100% within free limits.
    """
    auth_token = token or GITHUB_TOKEN or os.getenv("GITHUB_ACCESS_TOKEN")
    # No hardcoded repo fallback: a broadly-scoped token must never delete
    # releases in someone else's repo just because GITHUB_REPOSITORY is unset.
    target_repo = repo or GITHUB_REPOSITORY

    if not auth_token or not target_repo:
        print("[-] GITHUB_TOKEN or GITHUB_REPOSITORY is missing. Cannot perform release cleanup.")
        return {"deleted": 0, "preserved": 0, "freed_bytes": 0}

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    print("=" * 70)
    print(f"🧹 GITHUB RELEASES AUTO-CLEANUP (Threshold: {days} days)")
    print(f"[*] Target Repository: {target_repo}")
    print(f"[*] Dry Run: {dry_run}")
    print("=" * 70)

    url = f"https://api.github.com/repos/{target_repo}/releases"
    try:
        # Paginate through ALL releases. The API returns 30 per page by default;
        # the old single request meant that once >30 releases accumulated, the
        # oldest (the ones we actually need to delete) were never even seen.
        releases: List[Dict[str, Any]] = []
        page = 1
        while True:
            res = requests.get(url, headers=headers,
                               params={"per_page": 100, "page": page}, timeout=20)
            if res.status_code != 200:
                print(f"[-] Failed to fetch releases (page {page}): {res.status_code} - {res.text}")
                if page == 1:
                    return {"deleted": 0, "preserved": 0, "freed_bytes": 0}
                break
            batch = res.json()
            if not batch:
                break
            releases.extend(batch)
            if len(batch) < 100:
                break
            page += 1

        print(f"[*] Found {len(releases)} total release(s) in repository.")

        now = datetime.now(timezone.utc)
        deleted_count = 0
        preserved_count = 0
        total_freed_bytes = 0

        for rel in releases:
            rel_id = rel.get("id")
            tag_name = rel.get("tag_name")
            created_str = rel.get("created_at")
            # One malformed/missing timestamp must not abort the whole cleanup.
            if not created_str:
                print(f"[-] Release '{tag_name}' (ID: {rel_id}) has no created_at; skipping.")
                preserved_count += 1
                continue
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                print(f"[-] Release '{tag_name}' has unparseable created_at '{created_str}'; skipping.")
                preserved_count += 1
                continue
            age_days = (now - created_at).total_seconds() / 86400.0

            assets = rel.get("assets", [])
            asset_size = sum(a.get("size", 0) for a in assets)

            if age_days >= days:
                print(f"[!] Release '{tag_name}' (ID: {rel_id}) is {age_days:.1f} days old (>= {days} days).")
                print(f"    - Assets: {[a.get('name') for a in assets]} ({asset_size / (1024*1024):.2f} MB)")

                if not dry_run:
                    # 1. Delete release
                    del_res = requests.delete(f"{url}/{rel_id}", headers=headers, timeout=15)
                    if del_res.status_code == 204:
                        print(f"    [+] Deleted release {rel_id} successfully.")
                        deleted_count += 1
                        total_freed_bytes += asset_size

                        # 2. Delete git tag to free tag ref
                        tag_del_url = f"https://api.github.com/repos/{target_repo}/git/refs/tags/{tag_name}"
                        requests.delete(tag_del_url, headers=headers, timeout=15)
                    else:
                        print(f"    [-] Could not delete release {rel_id}: {del_res.status_code}")
                else:
                    print(f"    [Dry Run] Would delete release {rel_id} and free {asset_size / (1024*1024):.2f} MB.")
                    deleted_count += 1
                    total_freed_bytes += asset_size
            else:
                print(f"[+] Preserving release '{tag_name}' (Age: {age_days:.1f} days < {days} days). Still active for Buffer queue.")
                preserved_count += 1

        print("\n" + "=" * 70)
        print("✨ CLEANUP SUMMARY")
        print(f"- Releases Deleted: {deleted_count}")
        print(f"- Releases Preserved: {preserved_count}")
        print(f"- Storage Freed: {total_freed_bytes / (1024*1024):.2f} MB")
        print("=" * 70)

        return {
            "deleted": deleted_count,
            "preserved": preserved_count,
            "freed_bytes": total_freed_bytes
        }

    except Exception as e:
        print(f"[-] Error during release cleanup: {e}")
        return {"deleted": 0, "preserved": 0, "freed_bytes": 0}

def main():
    parser = argparse.ArgumentParser(
        description="Auto-clean GitHub Releases older than N days (default: 5 days)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=5,
        help="Number of days to keep video releases before auto-deleting (default: 5)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate cleanup without actually deleting releases."
    )

    args = parser.parse_args()
    clean_old_releases(days=args.days, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
