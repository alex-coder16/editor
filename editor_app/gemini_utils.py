from google import genai
from google.genai.types import (
    Tool,
    GenerateContentConfig,
    GoogleSearch,
    Part,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
    FunctionDeclaration,
) 
import os
import time
import uuid
import asyncio
import logging
from pathlib import Path
from django.core.files.uploadedfile import UploadedFile
from django.conf import settings
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# It's recommended to use environment variables for API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY") # Replace YOUR_API_KEY or set environment variable
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY":
    logger.warning("GOOGLE_API_KEY not set or using default placeholder.")
    # Optionally raise an error or exit if the key is mandatory
    # raise ValueError("GOOGLE_API_KEY is not configured.")

# Initialize the client globally (or manage it within your Django app lifecycle)
try:
    client = genai.Client(api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Google Genai Client: {e}")
    client = None # Indicate client initialization failure

# --- Storage Paths ---
TEMP_UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'temp_uploads')
PROCESSED_VIDEOS_DIR = os.path.join(settings.MEDIA_ROOT, 'processed_videos')
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

# --- File Handling Store ---
# In-memory store for uploaded file details (simple approach for now)
# Store {gemini_file_name: {'file': gemini_file_object, 'local_path': temp_path}}
# Using file.name (e.g., "files/abc123xyz") as the key
uploaded_files_store = {}

# --- Async Gemini File Handling (Adapted from code_samples.txt) ---

async def upload_to_gemini(file_path, display_name=None) -> File | None:
    """Uploads a file to Gemini asynchronously."""
    if not client:
        logger.error("Genai Client not initialized. Cannot upload file.")
        return None
    logger.info(f"Uploading {file_path} to Gemini...")
    try:
        # Use a display name if provided, otherwise default
        file_name = display_name or os.path.basename(file_path)
        # Use asyncio.to_thread for the blocking SDK call
        response = await asyncio.to_thread(
            client.files.create,
            file=open(file_path, "rb"), # Pass file handle
            display_name=file_name
        )
        logger.info(f"Uploaded file: {response.display_name} as {response.name}") # Use response.name
        return response
    except Exception as e:
        logger.error(f"Error uploading file {file_path} to Gemini: {e}", exc_info=True)
        return None
    finally:
        # Ensure the file handle is closed if opened here
        # Note: If passing an existing handle, closing should be managed outside
        pass # The 'with open' context manager handles closing

async def wait_for_file_active(file_name: str) -> bool:
    """Waits for a single Gemini file to become ACTIVE asynchronously."""
    if not client:
        logger.error("Genai Client not initialized. Cannot check file status.")
        return False
    logger.info(f"Waiting for file {file_name} to become active...")
    try:
        while True:
            # Use asyncio.to_thread for the blocking SDK call
            file = await asyncio.to_thread(client.files.get, name=file_name)
            if file.state.name == "ACTIVE":
                logger.info(f"File {file_name} is active.")
                return True
            elif file.state.name != "PROCESSING":
                logger.error(f"File {file_name} failed processing. State: {file.state.name}")
                return False
            logger.info(f"File {file_name} is processing...")
            await asyncio.sleep(5) # Wait 5 seconds before checking again
    except Exception as e:
        logger.error(f"Error checking file status for {file_name}: {e}", exc_info=True)
        return False

async def handle_uploaded_file(uploaded_file: UploadedFile) -> File | None:
    """
    Saves the uploaded file temporarily, uploads it to Gemini asynchronously,
    stores its info, and returns the Gemini File object.
    """
    temp_filepath = None # Initialize to ensure it's defined in case of early error
    try:
        # Create a unique temporary file path
        temp_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
        temp_filepath = os.path.join(TEMP_UPLOAD_DIR, temp_filename)

        # Save the uploaded file locally (synchronously, as file I/O is often blocking)
        logger.info(f"Temporarily saving uploaded file to: {temp_filepath}")
        with open(temp_filepath, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Upload to Gemini asynchronously
        gemini_file = await upload_to_gemini(temp_filepath, display_name=uploaded_file.name)

        if gemini_file:
            # Wait for the file to be active
            if await wait_for_file_active(gemini_file.name):
                 # Store file info (Gemini object and local temp path) using file.name as key
                uploaded_files_store[gemini_file.name] = {'file': gemini_file, 'local_path': temp_filepath}
                logger.info(f"Stored Gemini file info for: {gemini_file.name}")
                # Optional: Schedule deletion later
                return gemini_file
            else:
                logger.error(f"Gemini file {gemini_file.name} failed to become active.")
                # Attempt to delete the failed Gemini file
                try:
                    await asyncio.to_thread(client.files.delete, name=gemini_file.name)
                    logger.info(f"Deleted inactive Gemini file: {gemini_file.name}")
                except Exception as del_err:
                    logger.error(f"Failed to delete inactive Gemini file {gemini_file.name}: {del_err}")
                # Clean up local temp file
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                return None
        else:
            # Clean up local temp file if Gemini upload failed
            logger.error("Gemini upload failed, cleaning up local temp file.")
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return None

    except Exception as e:
        logger.error(f"Error handling uploaded file {uploaded_file.name}: {e}", exc_info=True)
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath) # Cleanup on error
        return None

# --- Tool Functions (Keep Synchronous for now, wrap calls in asyncio.to_thread if needed) ---
# Moviepy operations are CPU-bound and might block the event loop if run directly in async functions.
# It's often better to run them in a separate thread.

def find_audio_track(genre: str, mood: str) -> str:
    """
    Finds a suitable audio track path based on genre and mood.
    Searches within the predefined audio library.
    Args:
        genre (str): The desired genre (e.g., 'cinematic', 'upbeat', 'calm').
        mood (str): The desired mood (e.g., 'happy', 'suspenseful', 'relaxing').
    Returns:
        str: The absolute file path to a suitable audio track, or an error message starting with 'Error:'.
    """
    logger.info(f"Searching for audio: Genre='{genre}', Mood='{mood}'")
    audio_library_path = os.path.join(settings.BASE_DIR, 'static', 'audio_library')
    try:
        if not os.path.isdir(audio_library_path):
             raise FileNotFoundError(f"Audio library directory not found at {audio_library_path}")

        for filename in os.listdir(audio_library_path):
            # Simple matching logic (case-insensitive)
            if genre.lower() in filename.lower() or mood.lower() in filename.lower():
                # Check if it's a file (and optionally check extension)
                found_path = os.path.join(audio_library_path, filename)
                if os.path.isfile(found_path):
                    logger.info(f"Found audio track: {found_path}")
                    return found_path # Return the first match

        logger.warning("No suitable audio track found.")
        return "Error: No suitable audio track found matching the criteria."
    except FileNotFoundError as e:
        logger.error(f"Audio library search error: {e}")
        return str(e) # Return the specific error message
    except Exception as e:
        logger.error(f"Error searching audio library: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while searching for audio: {e}"

def send_video(edited_video_temp_path: str) -> str:
    """
    Moves the edited video from a temporary path to the final processed media directory
    and returns the final URL accessible to the user.
    Args:
        edited_video_temp_path (str): The temporary path where the edited video is stored.
    Returns:
        str: The final URL of the processed video, or an error message starting with 'Error:'.
    """
    logger.info(f"Preparing to send video from temporary path: {edited_video_temp_path}")
    if not os.path.exists(edited_video_temp_path):
        logger.error("Edited video temporary file not found.")
        return "Error: Edited video temporary file not found."

    try:
        # Create a unique name for the final file
        base_name = os.path.basename(edited_video_temp_path)
        # Ensure the base name doesn't have problematic characters if needed
        final_filename = f"processed_{uuid.uuid4()}_{base_name}"
        final_filepath = os.path.join(PROCESSED_VIDEOS_DIR, final_filename)

        # Move the file (os.rename is generally atomic on local filesystems)
        os.rename(edited_video_temp_path, final_filepath)
        logger.info(f"Moved edited video to final location: {final_filepath}")

        # Construct the URL (requires MEDIA_URL to be set correctly in settings.py)
        # Ensure no leading slash if MEDIA_URL already has one
        relative_path = os.path.join('processed_videos', final_filename).replace('\\', '/')
        final_url = os.path.join(settings.MEDIA_URL, relative_path).replace('\\', '/')
        # Handle potential double slashes if MEDIA_URL ends with / and relative_path starts with /
        final_url = final_url.replace('//', '/')
        if settings.MEDIA_URL.startswith('/') and not final_url.startswith('/'):
             final_url = '/' + final_url # Ensure leading slash if MEDIA_URL has one

        logger.info(f"Final video URL: {final_url}")
        return final_url # Return the URL for web access

    except Exception as e:
        logger.error(f"Error moving/saving final video: {e}", exc_info=True)
        # Attempt to clean up temp file if it still exists (it shouldn't after rename)
        if os.path.exists(edited_video_temp_path):
             logger.warning(f"Temporary file {edited_video_temp_path} still exists after expected move.")
             # os.remove(edited_video_temp_path) # Optional: attempt cleanup
        return f"Error: Failed to save or provide URL for the final video: {e}"

async def add_background_track(original_video_file_name: str, audio_track_path: str, start_time: int, end_time: int) -> File | None:
    """
    Adds an audio track to a video file between specified timestamps using moviepy.
    Uploads the edited video back to Gemini for review and saves a local copy via send_video.
    Runs moviepy operations in a separate thread to avoid blocking the event loop.
    Args:
        original_video_file_name (str): The Gemini name (e.g., 'files/abc123xyz') of the original video file.
        audio_track_path (str): The local file path to the audio track to add.
        start_time (int): The time in seconds where the background audio should start.
        end_time (int): The time in seconds where the background audio should end.
    Returns:
        File | None: The Gemini File object of the *edited* video uploaded for review, or None on failure.
    Raises:
        ValueError: If input parameters are invalid (e.g., file not found in store, times invalid).
        FileNotFoundError: If local video/audio files are missing.
        ConnectionError: If uploading or activating the edited video fails.
        Exception: For other processing errors (e.g., moviepy errors).
    """
    logger.info(f"Adding background track '{audio_track_path}' to video '{original_video_file_name}' from {start_time}s to {end_time}s")

    # 1. Get local path of the original video from our store
    if original_video_file_name not in uploaded_files_store:
        raise ValueError(f"Original video file name '{original_video_file_name}' not found in local store.")
    original_video_path = uploaded_files_store[original_video_file_name]['local_path']

    if not os.path.exists(original_video_path):
         raise FileNotFoundError(f"Original video file not found at temporary path: {original_video_path}")
    if not os.path.exists(audio_track_path):
        raise FileNotFoundError(f"Audio track file not found at path: {audio_track_path}")
    if end_time <= start_time:
        raise ValueError("End time must be greater than start time.")

    edited_video_temp_path = None # Initialize for cleanup

    try:
        # --- Moviepy operations in a separate thread ---
        def _run_moviepy_edit():
            nonlocal edited_video_temp_path # Allow modification of outer scope variable
            video_clip = None
            audio_clip = None
            final_audio = None
            try:
                video_clip = VideoFileClip(original_video_path)
                audio_clip = AudioFileClip(audio_track_path)

                edit_duration = end_time - start_time
                # Trim audio if longer than the target segment
                if audio_clip.duration > edit_duration:
                    audio_clip = audio_clip.subclip(0, edit_duration)

                # Set the start time for the audio within the composite
                audio_clip = audio_clip.set_start(start_time)

                # Composite audio
                original_audio = video_clip.audio
                if original_audio:
                    # Ensure original audio has a duration if needed (sometimes missing)
                    if original_audio.duration is None:
                         original_audio.duration = video_clip.duration
                    # Layer the new audio clip over the original
                    final_audio = CompositeAudioClip([original_audio.set_duration(video_clip.duration), audio_clip])
                else:
                    # If no original audio, use the new clip, potentially padded/looped if shorter than video
                    # For simplicity, just set duration - might need more complex logic for looping/padding
                    final_audio = audio_clip.set_duration(video_clip.duration)

                # Set the composite audio to the video
                video_clip = video_clip.set_audio(final_audio)

                # Save the edited video to a new temporary path
                base_original_name = os.path.basename(original_video_path)
                edited_filename = f"edited_{uuid.uuid4()}_{base_original_name}"
                _edited_video_temp_path = os.path.join(TEMP_UPLOAD_DIR, edited_filename)
                logger.info(f"Saving edited video temporarily to: {_edited_video_temp_path}")
                # Specify codecs for better compatibility
                video_clip.write_videofile(_edited_video_temp_path, codec='libx264', audio_codec='aac', threads=4, logger='bar') # Use multiple threads

                return _edited_video_temp_path # Return the path on success

            finally:
                # Ensure clips are closed to release file handles
                if video_clip: video_clip.close()
                if audio_clip: audio_clip.close()
                # CompositeAudioClip doesn't have a close method directly, its components are closed
                # if original_audio: original_audio.close() # Close original if it was loaded separately

        # Run the blocking moviepy code in a thread
        logger.info("Starting moviepy editing in background thread...")
        edited_video_temp_path = await asyncio.to_thread(_run_moviepy_edit)
        logger.info("Moviepy editing finished.")

        if not edited_video_temp_path or not os.path.exists(edited_video_temp_path):
             raise RuntimeError("Moviepy editing failed to produce an output file.")

        # 4. Upload edited video back to Gemini asynchronously
        logger.info("Uploading edited video back to Gemini for review...")
        edited_gemini_file = await upload_to_gemini(
            edited_video_temp_path,
            display_name=f"edited_{os.path.basename(original_video_path)}"
        )
        if not edited_gemini_file:
            raise ConnectionError("Failed to upload edited video to Gemini.")

        # 5. Wait for edited video to be active
        logger.info(f"Waiting for edited Gemini file {edited_gemini_file.name} to become active...")
        if not await wait_for_file_active(edited_gemini_file.name):
             logger.error(f"Edited video {edited_gemini_file.name} failed to process in Gemini.")
             # Attempt to delete the failed Gemini upload if possible
             try:
                 await asyncio.to_thread(client.files.delete, name=edited_gemini_file.name)
                 logger.info(f"Deleted inactive edited Gemini file: {edited_gemini_file.name}")
             except Exception as del_err:
                 logger.error(f"Also failed to delete unsuccessful Gemini upload {edited_gemini_file.name}: {del_err}")
             raise ConnectionError(f"Edited video {edited_gemini_file.name} failed to process in Gemini.")

        # Store info about the *edited* Gemini file
        uploaded_files_store[edited_gemini_file.name] = {'file': edited_gemini_file, 'local_path': edited_video_temp_path}
        logger.info(f"Stored edited Gemini file info for: {edited_gemini_file.name}")

        # 6. Call send_video (sync) to save a persistent local copy
        # Run this sync function in a thread as well if it involves significant I/O
        logger.info("Calling send_video to store final local copy...")
        final_video_location = await asyncio.to_thread(send_video, edited_video_temp_path)
        # Note: send_video moves the file, so edited_video_temp_path is no longer valid after this point.
        edited_video_temp_path = None # Mark as moved/handled
        logger.info(f"Final video stored locally at/via: {final_video_location}")

        # 7. Return the Gemini File object of the *edited* video
        logger.info(f"Returning edited Gemini file object: {edited_gemini_file.name}")
        return edited_gemini_file # Return the Gemini file object for AI review

    except Exception as e:
        logger.error(f"Error during add_background_track: {e}", exc_info=True)
        # Clean up temporary edited file if it exists and wasn't moved/handled
        if edited_video_temp_path and os.path.exists(edited_video_temp_path):
            logger.info(f"Cleaning up temporary edited file due to error: {edited_video_temp_path}")
            os.remove(edited_video_temp_path)
        # Re-raise the exception so the caller (and Gemini) knows the tool failed
        raise e


# --- Gemini Tool Definition (using google.genai.types) ---

find_audio_tool = FunctionDeclaration(
    name="find_audio_track",
    description="Searches the local audio library for a background music track based on genre and mood. Returns the full path to the audio file or an error message.",
    parameters={
        "type": "object",
        "properties": {
            "genre": {"type": "string", "description": "The desired genre (e.g., 'cinematic', 'upbeat', 'calm', 'electronic')."},
            "mood": {"type": "string", "description": "The desired mood (e.g., 'happy', 'suspenseful', 'relaxing', 'energetic')."}
        },
        "required": ["genre", "mood"]
    }
)

add_background_track_tool = FunctionDeclaration(
    name="add_background_track",
    description="Adds a specified audio track to a video between the start and end times using moviepy. The edited video file is then uploaded and returned for review.",
    parameters={
        "type": "object",
        "properties": {
            "original_video_file_name": {"type": "string", "description": "The Gemini name (e.g., 'files/abc123xyz') of the original video file provided previously."},
            "audio_track_path": {"type": "string", "description": "The full local file path to the audio track, usually obtained from 'find_audio_track'."},
            "start_time": {"type": "integer", "description": "The time in seconds within the video where the background audio should begin."},
            "end_time": {"type": "integer", "description": "The time in seconds within the video where the background audio should end."}
        },
        "required": ["original_video_file_name", "audio_track_path", "start_time", "end_time"]
    }
)

# Combine tools into a Tool object
video_editing_tool = Tool(
    function_declarations=[
        find_audio_tool,
        add_background_track_tool,
    ]
)

# --- Model Interaction Logic ---

# Store active chats (simple in-memory store)
# Key: A unique identifier for the chat session (e.g., user ID, session ID)
# Value: google.genai.models.Chat object
active_chats = {}

async def get_or_create_chat(session_id: str) -> Chat | None:
    """Gets an existing chat session or creates a new one."""
    if not client:
        logger.error("Genai Client not initialized. Cannot create chat.")
        return None

    if session_id in active_chats:
        return active_chats[session_id]
    else:
        logger.info(f"Creating new chat session for ID: {session_id}")
        try:
            # Define safety settings (adjust as needed)
            safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
            }
            # Define generation config
            generation_config = GenerateContentConfig(
                temperature=0.7, # Adjust creativity
                # max_output_tokens=8192, # Set limits if needed
                # top_p=1.0,
                # top_k=40
            )
            # System instruction explaining the tools and workflow
            system_instruction = (
                "You are an AI video editor assistant. "
                "The user will provide a video file and text instructions. "
                "Use the available tools to fulfill the user's editing requests. "
                "1. If the user asks for background music, use 'find_audio_track' first to get the audio file path based on genre/mood. "
                "2. Use 'add_background_track' to add the found audio (or user-specified audio path if they provide one) to the video. You MUST provide the Gemini file name (e.g., 'files/abc123xyz') of the original video, the audio path from step 1, and the start/end times. "
                "3. The 'add_background_track' tool will perform the edit and return the *new* Gemini File object representing the edited video. "
                "4. Review the result. If the edit seems correct based on the request, inform the user that the edit is complete and provide the final video URL (which is generated automatically when the tool succeeds). "
                "5. If the edit needs changes or the tool failed, explain the issue and ask the user for clarification or try the tool again with corrected parameters. "
                "Always refer to video files using their Gemini file name (e.g., 'files/abc123xyz') when calling tools."
            )

            # Create the chat using client.chats.create
            chat = await asyncio.to_thread(
                client.chats.create,
                model='gemini-1.5-flash-latest', # Or 'gemini-1.5-pro-latest'
                tools=[video_editing_tool],
                config=generation_config,
                safety_settings=safety_settings,
                system_instruction=system_instruction,
                history=[] # Start with empty history for a new chat
            )
            active_chats[session_id] = chat
            return chat
        except Exception as e:
            logger.error(f"Failed to create chat session for {session_id}: {e}", exc_info=True)
            return None

async def generate_response(session_id: str, prompt_parts: list[str | Part]) -> str:
    """
    Sends a prompt (text + files) to the Gemini chat, handles function calls, and returns the final text response.
    """
    if not client:
        return "Error: The AI backend client is not initialized."

    chat = await get_or_create_chat(session_id)
    if not chat:
        return "Error: Could not establish a chat session with the AI."

    logger.info(f"Sending prompt to chat {session_id}: {prompt_parts}")

    try:
        # Send the message parts (text, files)
        response = await asyncio.to_thread(chat.send_message, prompt_parts)

        # --- Function Calling Loop ---
        while response.function_calls and len(response.function_calls) > 0:
            logger.info(f"Received function calls: {[fc.name for fc in response.function_calls]}")
            function_response_parts = []

            # Execute functions (potentially in parallel if safe)
            # Note: For simplicity, executing sequentially here.
            # Consider asyncio.gather for parallel execution if functions are independent.
            tasks = []
            for func_call in response.function_calls:
                func_name = func_call.name
                args = func_call.args

                logger.info(f"Executing function: {func_name} with args: {args}")
                part = None # Initialize part for this function call
                try:
                    if func_name == "find_audio_track":
                        # This function is sync, run in thread
                        function_result = await asyncio.to_thread(
                            find_audio_track,
                            genre=args.get("genre"),
                            mood=args.get("mood")
                        )
                        # Return the path or error message as content
                        part = Part.from_function_response(
                            name=func_name,
                            response={"content": function_result}
                        )

                    elif func_name == "add_background_track":
                        # This function is async, await it directly
                        edited_file_object = await add_background_track(
                            original_video_file_name=args.get("original_video_file_name"),
                            audio_track_path=args.get("audio_track_path"),
                            start_time=args.get("start_time"),
                            end_time=args.get("end_time")
                        )
                        # Return the *edited* File object if successful
                        if edited_file_object:
                             part = Part.from_function_response(
                                 name=func_name,
                                 response={"file": edited_file_object} # Send the file object back
                             )
                        else:
                             # This case might occur if add_background_track returns None
                             # instead of raising an error on failure.
                             logger.error(f"Tool {func_name} completed but returned None.")
                             part = Part.from_function_response(
                                 name=func_name,
                                 response={"content": f"Error: Tool {func_name} failed internally and returned no result."}
                             )
                    else:
                        # Handle unknown function call
                        logger.error(f"Unknown function call received: {func_name}")
                        part = Part.from_function_response(
                            name=func_name,
                            response={"content": f"Error: Unknown function '{func_name}' requested."}
                        )

                except Exception as e:
                    # Catch errors during function execution (sync or async)
                    logger.error(f"Error executing function {func_name}: {e}", exc_info=True)
                    part = Part.from_function_response(
                        name=func_name,
                        response={"content": f"Error executing tool {func_name}: {e}"}
                    )

                if part:
                    function_response_parts.append(part)

            # Send the function responses back to the model
            if function_response_parts:
                logger.info(f"Sending function responses back to model: {function_response_parts}")
                response = await asyncio.to_thread(chat.send_message, function_response_parts)
            else:
                # Should not happen if there were function calls, but break defensively
                logger.warning("Function calls received but no response parts generated.")
                break

        # --- Extract Final Text Response ---
        # After the loop (or if no function calls), get the text response
        if response.text:
            logger.info(f"Received final text response from Gemini: {response.text}")
            return response.text
        else:
            # Handle cases where the final response might not be text
            # (e.g., if the model only returns function calls or stops)
            logger.warning("No final text content received from Gemini.")
            # Check if there's other content like candidates or parts
            if response.candidates and response.candidates[0].content.parts:
                 # Attempt to construct a string from parts if possible, otherwise return generic message
                 try:
                     return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')) or "AI response did not contain text."
                 except Exception:
                     return "Received a non-text response from the AI."
            return "AI response did not contain text."

    except Exception as e:
        logger.error(f"Error during chat communication with Gemini: {e}", exc_info=True)
        return f"Error: An error occurred while communicating with the AI: {e}"
