from typing_extensions import deprecated
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import logging

load_dotenv(".env.local")

# Silence HTTP/2 HPACK debug logs
logging.getLogger("hpack.hpack").setLevel(logging.WARNING)


'''
Note - was throwing error on docker build due env vars not
 read - .dockerignore file was excluding .env.local
 "RUN uv run src/agent.py download-files" run fails
'''
# SUPABASE_URL = os.environ.get("SUPABASE_URL")
# SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class _LazySupabase:
    _client: Client | None = None

    def _get(self) -> Client:
        if self._client is None:
            url = os.environ.get("SUPABASE_URL")
            key = os.environ.get("SUPABASE_KEY")
            if not url or not key:
                raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
            self._client = create_client(url, key)
        return self._client

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


supabase: Client = _LazySupabase()  # type: ignore[assignment]


def db_book_appointment(
    session_id: str,
    contact_number: str,
    date: str,
    time: str
):
    try:
        result = (
            supabase
            .table("appointments")
            .insert({
                "session_id": session_id,
                "contact_number": contact_number,
                "date": date,
                "time": time,
                "status": "BOOKED"
            })
            .execute()
        )

        return result.data[0]

    except Exception as e:
        # Unique constraint violation → slot already booked
        if "unique_active_slot" in str(e):
            return {"error": "SLOT_ALREADY_BOOKED"}
        raise

# Todo - deprecate after making available slots dynamic
def db_get_all_appointments():
    result = (
        supabase
        .table("appointments")
        .select("id, date, time, status")
        .eq("status", "BOOKED")
        .order("date", desc=False)
        .order("time", desc=False)
        .execute()
    )

    return result.data

def db_get_appointments(contact_number: str):
    result = (
        supabase
        .table("appointments")
        .select("id, date, time, status")
        .eq("contact_number", contact_number)
        .eq("status", "BOOKED")
        .order("date", desc=False)
        .order("time", desc=False)
        .execute()
    )

    return result.data

def db_cancel_appointment(appointment_id: str):
    result = (
        supabase
        .table("appointments")
        .update({ "status": "CANCELLED" })
        .eq("id", appointment_id)
        .execute()
    )

    if not result.data:
        return {"error": "APPOINTMENT_NOT_FOUND"}

    return result.data[0]

def db_modify_appointment(
    appointment_id: str,
    new_date: str,
    new_time: str
):
    try:
        result = (
            supabase
            .table("appointments")
            .update({
                "date": new_date,
                "time": new_time
            })
            .eq("id", appointment_id)
            .execute()
        )

        if not result.data:
            return {"error": "APPOINTMENT_NOT_FOUND"}

        return result.data[0]

    except Exception as e:
        if "unique_active_slot" in str(e):
            return {"error": "SLOT_ALREADY_BOOKED"}
        raise

def save_call_summary(session_id: str, contact_number: str, summary: str):
    """
    Persists the call summary with timestamp.
    """

    result = supabase.table("call_summaries").insert({
        "session_id": session_id,
        "contact_number": contact_number,
        "summary": summary
    }).execute()

    return result.data[0]