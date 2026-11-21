from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "db"
OUT = ROOT / "outputs"
TEST_INPUT = ROOT/'test_input.csv'
SEED = 42

WEEK = pd.Timedelta(7, "days")
GAP = pd.Timedelta(7, "days")

DATA_START = pd.Timestamp("2026-08-29 00:00:00")
TEST_WEEK = pd.Timestamp("2026-11-21 16:00:00")
DATA_END = TEST_WEEK - GAP

VAL_WEEKS = pd.date_range(end=DATA_END - WEEK, periods=2, freq=WEEK)
TRAIN_WEEKS = pd.date_range(end=VAL_WEEKS[0] - WEEK, periods=6, freq=WEEK)

HIDDEN_DIM = 64
GNN_LAYERS = 2
EPOCHS = 40
LR = 5e-3
EVAL_EVERY=1

SCHEMA = {
    "Users": {
        "time": "first_seen_at",
        "bool": ["is_bot", "is_cohort"],
    },
    "Repositories": {
        "time": "first_seen_at",
    },
    "PullRequests": {
        "time": "first_seen_at",
        "known_later": {"merged_at": ["merged_at", "merged_by_id"], "closed_at": ["closed_at"]},
        "fks": {"repo_id": "Repositories", "author_id": "Users", "merged_by_id": "Users"},
        "bool": ["draft", "is_cross_repo"],
    },
    "Issues": {
        "time": "first_seen_at",
        "known_later": {"closed_at": ["closed_at", "state_reason"]},
        "fks": {"repo_id": "Repositories", "author_id": "Users",
                "pull_request_id": "PullRequests", "assignee_id": "Users"},
        "bool": ["is_pull_request", "locked"],
    },
    "IssueComments": {
        "time": "created_at",
        "fks": {"issue_id": "Issues", "repo_id": "Repositories", "author_id": "Users"},
        "bool": ["is_pull_request"],
    },
    "PRReviews": {
        "time": "first_seen_at",
        "known_later": {"dismissed_at": ["dismissed_at"]},
        "fks": {"pr_id": "PullRequests", "repo_id": "Repositories", "reviewer_id": "Users"},
    },
    "PRReviewComments": {
        "time": "created_at",
        "fks": {"pr_id": "PullRequests", "repo_id": "Repositories", "author_id": "Users",
                "pull_request_review_id": "PRReviews", "in_reply_to_id": "PRReviewComments"},
    },
    "Pushes": {
        "time": "created_at",
        "fks": {"user_id": "Users", "repo_id": "Repositories"},
        "bool": ["is_new_ref"],
    },
}


FEATURES = {
    "Repositories": {
        "present": ["org_id"],
    },
    "PullRequests": {
        "present": ["merged_at", "closed_at"],
    },
    "Issues": {
        "present": ["assignee_id", "closed_at", "created_via_app"],
    },
    "IssueComments": {
        "present": ["via_app"],
    },
    "PRReviews": {
        "present": ["dismissed_at"],
        "category": {"state": ["approved", "commented", "changes_requested"]},
    },
    "PRReviewComments": {
        "present": ["in_reply_to_id"],
        "category": {"subject_type": ["line", "file"]},
    },
    "Pushes": {
        "category": {"ref_type": ["branch", "tag"]},
    },
}

FEATURE_GROUPS = ["time", "bools", "present", "category"]


ACTIVITY = {
    "open_pr": ("PullRequests", "author_id", "created_at"),
    "open_issue": ("Issues", "author_id", "created_at"),
    "issue_comment": ("IssueComments", "author_id", "created_at"),
    "pr_review": ("PRReviews", "reviewer_id", "submitted_at"),
    "pr_review_comment": ("PRReviewComments", "author_id", "created_at"),
    "push": ("Pushes", "user_id", "created_at"),
}

TARGETS = list(ACTIVITY)