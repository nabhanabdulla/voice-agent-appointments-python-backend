from datetime import datetime
from functools import wraps
import json
import logging
from typing import Annotated

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import function_tool, RunContext
from onnxruntime.capi.onnxruntime_inference_collection import Session
from pydantic import BaseModel


logger = logging.getLogger("agent")

load_dotenv(".env.local")

session_state = {
    "user_identified": False,
    "contact_number": None
}

appointment_slots = [
    {
        "slot_id": "slot_1",
        "date": "2026-01-22",
        "time": "10:00",
        "status": "AVAILABLE"
    },
    {
        "slot_id": "slot_2",
        "date": "2026-01-22",
        "time": "14:00",
        "status": "AVAILABLE"
    },
    {
        "slot_id": "slot_3",
        "date": "2026-01-23",
        "time": "11:00",
        "status": "AVAILABLE"
    }
]

# class SessionState:
#     def __init__(self):
#         self.user_identified = False
#         self.contact_number = None

# SESSION_STATE = SessionState()

# SESSION_STATE: dict[str, SessionState] = {}

# def get_session(room_id: str) -> SessionState:
#     if room_id not in SESSION_STATE:
#         SESSION_STATE[room_id] = SessionState()
#     return SESSION_STATE[room_id]

TOOL_REQUIREMENTS = {
    "identify_user": [],
    "fetch_slots": ["user_identified"],
    "book_appointment": ["user_identified"],
    "retrieve_appointments": ["user_identified"],
    "cancel_appointment": ["user_identified"],
    "modify_appointment": ["user_identified"],
    "end_conversation": []
}


# TODO - Fix this - tool calls failing for identify_user (not seeing error also)
def dispatch(tool_name: str):
    def decorator(tool_fn):
        @wraps(tool_fn)
        async def wrapper(*args, **kwargs):
            ctx = kwargs.get("ctx")
            if ctx is None:
                return {
                    "error": "MISSING_CONTEXT",
                    "message": "LiveKit context not found."
                }

            room_id = ctx.room.name
            session = get_session(room_id)

            print(f"Dispater called: room_id - {room_id}")

            # Enforce requirements
            if "user_identified" in TOOL_REQUIREMENTS.get(tool_name, []):
                if not session.user_identified or not session.contact_number:
                    output =  {
                        "error": "USER_NOT_IDENTIFIED",
                        "message": "User must be identified before this action."
                    }

                    logger.warning(
                        "[TOOL BLOCKED] %s | room=%s | output=%s",
                        tool_name,
                        room_id,
                        output
                    )

                    return output

            # Call the actual tool
            result = await tool_fn(*args, **kwargs)
            logger.info(
                "[TOOL RESULT] %s | room=%s | output=%s",
                tool_name,
                room_id,
                json.dumps(result, default=str)
            )

            return result

        return wrapper
    return decorator

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an AI voice assistant for booking and managing appointments.
You speak in a calm, friendly, professional tone suitable for phone calls.

Your primary goal is to help users:
- Book appointments
- Retrieve existing appointments
- Modify or cancel appointments
- End the conversation politely

You must follow the rules below strictly.

────────────────────────
GENERAL BEHAVIOR
────────────────────────
- Respond in clear, concise spoken English.
- Keep responses short and natural for voice.
- Do not use markdown, emojis, or lists in spoken responses.
- Do not mention internal systems, tools, or databases.
- Maintain conversation context across turns.
- Handle one user request at a time.

────────────────────────
USER IDENTIFICATION
────────────────────────
- A user is uniquely identified by their phone number.
- If the user has not been identified yet and requests any appointment action,
  you MUST ask for their phone number first.
- Do not proceed with booking, retrieving, modifying, or cancelling appointments
  until the user is identified.

────────────────────────
TOOL CALLING RULES
────────────────────────
- You may call ONLY the tools provided to you.
- If a tool call is required, respond ONLY with a valid JSON tool call.
- Do NOT include any spoken text when making a tool call.
- After a tool response, continue the conversation naturally.

────────────────────────
AVAILABLE TOOLS
────────────────────────
1. identify_user(contact_number)
2. fetch_slots()
3. book_appointment(date, time)
4. retrieve_appointments()
5. cancel_appointment(appointment_id)
6. modify_appointment(appointment_id, new_date, new_time)
7. end_conversation()

────────────────────────
DATE AND TIME EXTRACTION
────────────────────────
- Always extract dates and times explicitly from user speech.
- Convert all dates to ISO format: YYYY-MM-DD.
- Convert all times to 24-hour format: HH:MM.
- If a date or time is ambiguous or missing, ask a clarifying question
  instead of guessing.

────────────────────────
BOOKING RULES
────────────────────────
- Before booking, ensure the user is identified.
- Fetch and share available slots for user to book. Do not ask or allow user to book random slots not available in the fetched list of slots.
- If a requested slot is unavailable, inform the user politely and
  ask them to choose another slot.
- Always verbally confirm booking details after a successful booking.

────────────────────────
MODIFICATION AND CANCELLATION
────────────────────────
- If multiple appointments exist and the user does not specify which one,
  ask a clarifying question.
- Always confirm the final state after modification or cancellation.

────────────────────────
END OF CONVERSATION
────────────────────────
- When the user indicates they are done or says goodbye,
  call the end_conversation tool.
- End politely and professionally.

────────────────────────
ERROR HANDLING
────────────────────────
- If you are unsure what the user wants, ask a brief clarifying question.
- Never hallucinate appointments, dates, or confirmations.
- Never assume user intent without confirmation.

You must always prioritize correctness, clarity, and a natural voice experience.
""",
        )
        
        # TODO - Not getting picked up - figure out why (or not)
        # user_identified: bool = False
        # contact_number: str | None = None
        # appointment_slots = [
        #     {
        #         "slot_id": "slot_1",
        #         "date": "2026-01-22",
        #         "time": "10:00",
        #         "status": "AVAILABLE"
        #     },
        #     {
        #         "slot_id": "slot_2",
        #         "date": "2026-01-22",
        #         "time": "14:00",
        #         "status": "AVAILABLE"
        #     },
        #     {
        #         "slot_id": "slot_3",
        #         "date": "2026-01-23",
        #         "time": "11:00",
        #         "status": "AVAILABLE"
        #     }
        # ]

    @function_tool
    async def identify_user(
        self, 
        context: RunContext, 
        contact_number: Annotated[
            str,
            "User phone number used to identify or create a user record. Must be digits only. If the user states a phone number using words (for example: 'nine eight zero one two four three eight zero one zero'), it is converted into corresponding digits (for example: '98012438010')."
        ]
    ) -> dict:
        """
        Identifies the user for the current session using their phone number.
        Creates a new user record if one does not already exist.
        """

        # Example: normalize input
        # normalized_number = contact_number.strip()

        # Example: session / DB logic (pseudo-code)
        # user = db.get_user_by_contact(normalized_number)
        # if not user:
        #     user = db.create_user(normalized_number)
        # session = get_session(context.room.name)
        session_state["user_identified"] = True
        session_state["contact_number"] = contact_number

        return {
            "status": "identified",
            "contact_number": contact_number
        }

    @function_tool
    async def fetch_slots(
        self, 
        context: RunContext
    ) -> dict:
        """
        Fetches available appointment slots that the user can choose from.
        """
        return {
            "slots": appointment_slots
        }
        
    @function_tool
    async def book_appointment(
        self, 
        context: RunContext, 
        date: Annotated[
            str,
            "Appointment date in ISO format (YYYY-MM-DD). Example: 2026-01-22"
        ],
        time: Annotated[
            str,
            "Appointment time in 24-hour format (HH:MM). Example: 14:00"
        ]
    ) -> dict:
        """
        Books an appointment for the identified user.
        Prevents double-booking and validates slot availability.
        """

        # ─────────────────────────────
        # 1. Hard guard: user identity
        # ─────────────────────────────
        if not session_state["user_identified"] or not session_state["contact_number"]:
            return {
                "error": "USER_NOT_IDENTIFIED",
                "message": "User must be identified before booking an appointment."
            }

        # ─────────────────────────────
        # 2. Validate input format
        # ─────────────────────────────
        try:
            appointment_dt = datetime.fromisoformat(f"{date}T{time}")
        except ValueError:
            return {
                "error": "INVALID_DATE_TIME",
                "message": "Date or time format is invalid."
            }

        # ─────────────────────────────
        # 3. Check slot exists
        # ─────────────────────────────
        slot_exists = any(
            s["date"] == date and s["time"] == time and s["status"] == "AVAILABLE"
            for s in appointment_slots
        )

        if not slot_exists:
            return {
                "error": "SLOT_NOT_AVAILABLE",
                "message": "The requested slot does not exist."
            }

        # ─────────────────────────────
        # 4. Prevent double-booking
        # ─────────────────────────────
        # Example DB check (pseudo-code)
        #
        # existing = db.find_appointment_by_datetime(date, time)
        # if existing and existing.status == "booked":
        #     return {
        #         "error": "SLOT_ALREADY_BOOKED",
        #         "message": "That slot is already booked."
        #     }

        # ─────────────────────────────
        # 5. Create appointment
        # ─────────────────────────────
        appointment_id = f"apt_{int(appointment_dt.timestamp())}"

        for s in appointment_slots:
            if s["date"] == date and s["time"] == time:
                s["status"] = "BOOKED"

        # Example DB insert (pseudo-code)
        #
        # db.create_appointment(
        #     id=appointment_id,
        #     contact_number=session.contact_number,
        #     date=date,
        #     time=time,
        #     status="booked"
        # )

        # ─────────────────────────────
        # 6. Success response
        # ─────────────────────────────
        logger.info(f"Slots: {json.dumps(appointment_slots, indent=2)}")
        return {
            "status": "CONFIRMED",
            # "appointment_id": appointment_id,
            "date": date,
            "time": time,
            "contact_number": session_state["contact_number"]
        }

    

    @function_tool
    async def end_conversation(
        self, 
        context: RunContext
    ) -> dict:
        """
        Ends the current conversation after all actions are completed.
        """
        # await context.session.say("Thank you for calling. Have a great day!")
        
        # await context.session.room.disconnect()

        # Todo - figure out why llm response is sent back even after shutdown - due to graceful shutdown?
        context.session.shutdown()

        return {
            "status": "conversation_ended",
            "reason": "user_requested"
        }

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
