"""
Firebase Admin SDK utilities for Trade Tracker.
Connects to Firestore and fetches user trade data for analysis.
"""
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Firebase Admin SDK (only once)
_firebase_initialized = False

def init_firebase():
    """Initialize Firebase Admin SDK if not already done."""
    global _firebase_initialized
    if _firebase_initialized:
        return

    # Use absolute path based on this script's location
    # This ensures it finds the file regardless of where 'python main.py' is run from
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
    
    # If the env path is already absolute, use it. Otherwise, assume it's in the market-assistant root folder
    if os.path.isabs(env_path):
        cred_path = env_path
    else:
        # Strip any './' from the env path string if present to cleanly join
        clean_env_path = env_path.replace('./', '')
        cred_path = os.path.join(base_dir, clean_env_path)

    if not os.path.exists(cred_path):
        raise FileNotFoundError(
            f"Firebase credentials file not found at: {cred_path}\n"
            "Download it from Firebase Console → Project Settings → Service Accounts → Generate new private key"
        )

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    _firebase_initialized = True


def get_db():
    """Get Firestore client, initializing Firebase if needed."""
    init_firebase()
    return firestore.client()


def fetch_trades(user_id: str, start_date: str, end_date: str) -> list[dict]:
    """
    Fetch trades from Firestore for a given user and date range.

    Args:
        user_id: Firebase Auth UID of the user
        start_date: Start date string in YYYY-MM-DD format
        end_date: End date string in YYYY-MM-DD format

    Returns:
        List of trade dictionaries with all relevant fields
    """
    db = get_db()

    # Convert date strings to datetime objects for Firestore comparison
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

    # Primary query: filter by 'createdAt' (when the trade doc was saved)
    trades_ref = db.collection("trades")
    query = (
        trades_ref
        .where(filter=FieldFilter("userId", "==", user_id))
        .where(filter=FieldFilter("createdAt", ">=", start_dt))
        .where(filter=FieldFilter("createdAt", "<=", end_dt))
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
    )

    docs = list(query.stream())

    # Fallback: if createdAt query returns nothing, try filtering by 'buyDate'
    # This handles cases where the trade's buy date differs from its creation date
    if not docs:
        query_by_buy = (
            trades_ref
            .where(filter=FieldFilter("userId", "==", user_id))
            .where(filter=FieldFilter("buyDate", ">=", start_dt))
            .where(filter=FieldFilter("buyDate", "<=", end_dt))
            .order_by("buyDate", direction=firestore.Query.DESCENDING)
        )
        docs = list(query_by_buy.stream())

    trades = []
    for doc in docs:
        data = doc.to_dict()

        # Convert Firestore Timestamps to readable strings
        def timestamp_to_str(ts):
            if ts is None:
                return None
            if hasattr(ts, "strftime"):
                return ts.strftime("%Y-%m-%d")
            # Firestore Timestamp objects
            try:
                return ts.isoformat()[:10]
            except Exception:
                return str(ts)

        trade = {
            "id": doc.id,
            "stockName": data.get("stockName", ""),
            "buyPrice": data.get("buyPrice", 0),
            "sellPrice": data.get("sellPrice", 0),
            "quantity": data.get("quantity", 0),
            "brokerage": data.get("brokerage", 0),
            "taxes": data.get("taxes", 0),
            "tradeType": data.get("tradeType", "intraday"),
            "buyDate": timestamp_to_str(data.get("buyDate")),
            "sellDate": timestamp_to_str(data.get("sellDate")),
            "createdAt": timestamp_to_str(data.get("createdAt")),
            "mistakesMade": data.get("mistakesMade", []),
            "lessonsLearned": data.get("lessonsLearned", ""),
        }
        trades.append(trade)

    return trades
