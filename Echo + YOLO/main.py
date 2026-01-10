import os
import re
import logging
import time
import subprocess
from dotenv import load_dotenv
import speech_recognition as sr
from langchain_ollama import ChatOllama, OllamaLLM

# from langchain_openai import ChatOpenAI # if you want to use openai
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# importing tools
from tools.time import get_time, get_current_date # added "get_current_date from time.py 17/11/25"
from tools.OCR import read_text_from_latest_image
from tools.arp_scan import arp_scan_terminal
from tools.duckduckgo import duckduckgo_search_tool
from tools.matrix import matrix_mode
from tools.screenshot import take_screenshot
from tools.memory import remember_fact, recall_fact, list_all_memories, forget_fact, load_memory
from tools.todo import add_task, list_tasks, complete_task, delete_task, clear_completed_tasks
from tools.calculator import calculate

# Vision imports
from vision_system import VisionSystem
from shared.state import SharedState
from tools.face_recognition_tool import FaceRecognitionTools
from tools.profile_manager import ProfileManager

load_dotenv()

MIC_INDEX = None
TRIGGER_WORD = "echo"
CONVERSATION_TIMEOUT = 30  # seconds of inactivity before exiting conversation mode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Suppress ALSA warnings (from PyAudio) is hard in python, but we can set energy threshold manually
TRIG_WORD = "echo" # normalized
  # logging


recognizer = sr.Recognizer()
recognizer.pause_threshold = 2.0 # Wait 2 seconds of silence before stopping
recognizer.dynamic_energy_threshold = True
mic = sr.Microphone(device_index=MIC_INDEX)

# Initialize LLM (changed model from qwen3:1.7b to llama3.2b for more power and better tool calling "17/11/25")
llm = ChatOllama(model="llama3.2:3b", keep_alive="24h", reasoning=False)
profile_manager = ProfileManager()

# llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, organization=org_id) for openai

# Tool list (updated to account for the change in time.py 17/11/25)
tools = [get_time, get_current_date, arp_scan_terminal, read_text_from_latest_image, duckduckgo_search_tool, matrix_mode, take_screenshot, remember_fact, recall_fact, list_all_memories, forget_fact, calculate, add_task, list_tasks, complete_task, delete_task, clear_completed_tasks]
# Tool-calling prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Echo. You are NOT a formal assistant. You are a cool, witty, and street-smart best friend. "
            "You hang out with the user. You see what they see. "
            
            "## YOUR VIBE:\n"
            "- **Super Casual**: Talk like a real person texting a friend. Use slang (cool, sick, bet, no worries).\n"
            "- **Concise**: Don't drone on. Keep it punchy.\n"
            "- **Witty**: Be funny, sarcastic (playfully), and have an opinion.\n"
            "- **No Robot Speak**: NEVER say 'How can I assist you' or 'Is there anything else'. Say 'What's good?' or 'What's next?'.\n"
            
            "## ANTI-PATTERNS (DO NOT DO THIS):\n"
            "- ❌ Do not be overly polite or supportive ('I understand how you feel...'). That's boring.\n"
            "- ❌ Do not write long paragraphs unless asked.\n"
            "- ❌ Do not use formal grammar all the time.\n"
            
            "## VISION CONTEXT:\n"
            "- You can SEE. If the user input starts with 'Visual Context:', that's what you see.\n"
            "- React to it naturally. 'Yo, is that a cat?', 'Nice setup.'\n"
            "- If you recognize a face (e.g. 'Selim is here'), GREET THEM BY NAME! 'Yo Selim, what's good?'\n"
            
            "## FACE RECOGNITION:\n"
            "- If asked 'Who am I?', use 'recognize_face'.\n"
            "- If asked 'Remember me', use 'register_face'.\n"
            "- If asked 'Who do you know?', use 'list_known_people'.\n"
            
            "CRITICAL RULES: "
            "- When users ask 'what day is it', 'what's the date', or 'what time is it' (without specifying a city), you MUST call the get_current_date tool. "
            "- When users ask for time in a specific city like 'what time is it in London', call the get_time tool with that city name. "
            "- NEVER respond about time or date without calling one of these tools first. "
            "- When users ask for a calculation, use the 'calculate' tool. "
            "- MEMORY RULES: "
            "- Use 'remember_fact' to save info. Identify the subject (User, Mom, Dad, Brother, Dog, Best Friend, etc.) and pass it as the 'profile' argument (lowercase). Default to 'user' if about the speaker. "
            "- Use 'recall_fact' to retrieve info. Specify the 'profile' if asking about someone else (e.g., 'what is my mom's name?' -> profile='mom', 'what is my best friend's name?' -> profile='best friend'). DO NOT PASS A 'VALUE' ARGUMENT TO RECALL_FACT. "
            "- Use 'list_all_memories' to show what you know. You can specify a profile to filter. "
            "- Use 'forget_fact' to delete specific memories for a profile. "
            "- TO-DO LIST RULES: "
            "- Use 'add_task' to add new tasks. You can specify priority (high/medium/low) and due date. "
            "- Use 'list_tasks' to see what needs to be done. You can filter by status='incomplete' (default) or 'all', and by priority. "
            "- Use 'complete_task' to mark a task as done. "
            "- Use 'delete_task' to remove a task entirely. "
            "- Use 'clear_completed_tasks' to remove all finished tasks. "
            "- After calling a tool, give a quick, casual confirmation. 'Done.', 'Easy.', 'Got it.'",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# Agent + executor
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# TTS setup
def speak_text(text: str):
    try:
        subprocess.run(["espeak", text])
    except Exception as e:
        logging.error(f"❌ TTS failed: {e}")


def solve_simple_math(text: str):
    """
    Attempts to solve simple math problems locally using regex and eval.
    Returns the result string if successful, or None if it's not a simple math problem.
    """
    # Normalize text: replace "x" or "times" with "*", "plus" with "+", "minus" with "-", "divided by" with "/"
    text = text.lower()
    text = text.replace("times", "*").replace(" x ", " * ").replace("plus", "+").replace("minus", "-").replace("divided by", "/")
    
    # Regex to find simple math expressions like "5 + 5", "10 * 20", "100 / 5"
    # Looks for: number, operator, number
    match = re.search(r'(\d+(\.\d+)?)\s*([\+\-\*\/])\s*(\d+(\.\d+)?)', text)
    
    if match:
        try:
            expression = match.group(0)
            result = eval(expression)
            # Check if result is an integer (e.g. 5.0 -> 5)
            if result == int(result):
                result = int(result)
            return f"The result is {result}"
        except:
            return None
    return None

def check_local_intent(text: str):
    """
    Checks for common memory queries and handles them locally to bypass the LLM.
    """
    text = text.lower().strip(".,!?")
    
    # Pattern 1: "what/who is my [key]" -> profile="user" OR profile=[key]
    # e.g. "what is my name" -> key="name", profile="user"
    # e.g. "who is my best friend" -> key="best friend" -> profile="best friend"
    match = re.search(r"(?:what|who)(?:'s| is) my ([\w\s]+)$", text)
    if match:
        key = match.group(1).strip()
        
        # Check if the key is actually a profile (e.g., "mom", "dad", "best friend")
        memory = load_memory()
        if key in memory:
            logging.info(f"🧠 Local memory check: detected profile='{key}'")
            return list_all_memories.invoke({"profile": key})
            
        logging.info(f"🧠 Local memory check: key='{key}', profile='user'")
        return recall_fact.invoke({"key": key, "profile": "user"})
        
    # Pattern 2: "what/who is my [profile]'s [key]"
    # e.g. "what is my mom's name", "who is my best friend's name" (awkward but possible)
    match = re.search(r"(?:what|who)(?:'s| is) my ([\w\s]+)'s (\w+)$", text)
    if match:
        profile = match.group(1).strip()
        key = match.group(2).strip()
        logging.info(f"🧠 Local memory check: key='{key}', profile='{profile}'")
        return recall_fact.invoke({"key": key, "profile": profile})
        
    # Pattern 3: Todo List checks
    # e.g. "what is on my list", "show my tasks", "list tasks"
    todo_triggers = ["what is on my list", "what's on my list", "show my tasks", "list tasks", "show tasks", "my todo list"]
    if any(trigger in text for trigger in todo_triggers):
        logging.info("📝 Local todo check: detected list request")
        return list_tasks.invoke({"status": "incomplete"})

    return None

def check_vision_intent(text: str, shared_state: SharedState):
    """
    Checks for vision-related queries to answer immediately from shared state.
    """
    text = text.lower().strip(".,!?")
    
    # Common questions about what is being seen
    vision_triggers = [
        "what do you see", 
        "what can you see", 
        "what is in front of you",
        "describe the scene",
        "who is there",
        "is there anyone there",
        "what is this",
        "identify this",
        "what's this",
        "what am i holding"
    ]
    
    if any(trigger in text for trigger in vision_triggers):
        logging.info("👁️ Local vision check: detected vision query")
        summary = shared_state.get_vision_summary()
        
        # Smart Hint: If user seems to be asking about an object ("this") 
        # but we only see a person, suggest teaching.
        asking_about_object = any(x in text for x in ["what is this", "identify", "holding", "what's this"])
        only_sees_person = "person" in summary.lower() and "cell phone" not in summary.lower() and "cup" not in summary.lower() and "laptop" not in summary.lower() # Crude check, can be better
        
        # Better check: count objects in summary? 
        # Let's just append the hint if it looks like we might be missing something.
        if asking_about_object and "learn" not in text:
            summary += "\n(If I don't know what it is, say 'Learn this as [name]' to teach me!)"
            
        return summary
        
    return None

def check_learning_intent(text: str, vision_system: VisionSystem):
    """
    Checks if user wants to teach a new object.
    Format: "learn this [as] [name]" or "this is [a/my] [name]"
    """
    text = text.lower().strip(".,!?")
    
    # Simple regex for "learn this as [name]"
    # e.g. "Echo, learn this as my magic potion"
    match = re.search(r"learn this (?:as )?(.+)", text)
    if match:
        name = match.group(1).strip()
        logging.info(f"🎓 Learning request for: {name}")
        
        # Trigger learning
        success = vision_system.learn_current_object(name)
        if success:
            return f"Got it! I've learned what '{name}' looks like."
        else:
            return "I couldn't get a good look at it. Please hold it in the center of the frame and try again."

    return None

def check_identity_intent(text: str, face_tools: FaceRecognitionTools):
    """
    Checks if user is introducing themselves to trigger face registration.
    Format: "i am [name]", "my name is [name]", "remember me as [name]".
    """
    text = text.lower().strip(".,!?")
    
    # Regex for introduction
    triggers = [
        r"^i am ([a-z\s]+)$",
        r"^i'm ([a-z\s]+)$",
        r"^my name is ([a-z\s]+)$",
        r"^remember me as ([a-z\s]+)$",
        r"^this is ([a-z\s]+)$"
    ]
    
    for pattern in triggers:
        match = re.search(pattern, text)
        if match:
            name = match.group(1).strip()
            # Filter out common false positives if "name" is too long or a sentence
            if len(name.split()) > 3: 
                continue
                
            # Normalize common mishearings for "Selim"
            mishearings = ["sale", "saleem", "celim", "calim", "salim", "celine"]
            if name in mishearings or any(x in name for x in mishearings):
                name = "Selim"
                
            logging.info(f"👤 Identity trigger for: {name}")
            return face_tools.register_face(name)
            
    return None

def check_who_am_i_intent(text: str, face_tools: FaceRecognitionTools):
    """
    Checks for direct identity questions to bypass LLM.
    "who am i", "do you know me", "who is this"
    """
    text = text.lower().strip(".,!?")
    triggers = ["who am i", "who is this", "do you know me", "who do you see"]
    
    if any(trigger in text for trigger in triggers):
        logging.info("👤 Local identity check: detected 'who am i'")
        # Using recognize_face tool method directly return string
        return face_tools.recognize_face()
    
    return None

# Initialize Vision System
shared_state = SharedState()
vision = VisionSystem(shared_state)
face_tools = FaceRecognitionTools(vision)
vision.start()

# Verify Camera
# Temporarily skip camera verification - main loop will handle it
# if vision.check_camera():
#     logging.info("✅ Camera verification successful.")
#     # vision.start() # Already called after face_tools instantiation
# else:
#     logging.error("❌ Camera verification failed.")
#     # We can't speak here easily because 'speak_text' is defined above but designed to be used; 
#     # we should use it to notify user.
#     # Note: speak_text is defined at line 80.
#     # We will invoke it.
#     try:
#         subprocess.run(["espeak", "Warning. Camera not detected. Vision capabilities are disabled."])
#     except:
#         pass

# Initialize Agent Executor initially (default to Unknown)
current_agent_user = None
executor = None

def update_agent_for_user(user_name: str, tools_list: list):
    """Re-initializes the agent with the specific user's profile and permissions."""
    global executor, current_agent_user
    
    if user_name is None:
        user_name = "Unknown"
        
    logging.info(f"🔄 Switching profile to: {user_name}")
    
    # Load Profile
    profile = profile_manager.get_profile(user_name)
    
    # Get System Prompt
    system_prompt = profile_manager.get_system_prompt(profile)
    
    # Filter Tools
    allowed_tools = profile_manager.filter_tools(tools_list, profile)
    
    # Create Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create Agent
    agent = create_tool_calling_agent(llm, allowed_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=allowed_tools, verbose=True)
    current_agent_user = user_name
    
    logging.info(f"✅ Profile loaded: {user_name} ({profile.get('role', 'unknown')})")

# Initial load as Unknown
update_agent_for_user(None, tools)

# Main interaction loop
def write():
    global current_agent_user
    conversation_mode = False
    last_interaction_time = None

    # Profile Switching State
    last_person_time = time.time()
    
    try:
        with mic as source:
            logging.info("🎤 Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            # Manual threshold adjustment if auto is failing
            # recognizer.energy_threshold = 300  # Uncomment if too insensitive
            # recognizer.dynamic_energy_threshold = True
            
            logging.info(f"🔊 Energy threshold set to: {recognizer.energy_threshold}")
            
            while True:
                # 0. Check for User Change (Dynamic Profile Switching)
                visible_person = shared_state.person_name
                person_detected = shared_state.is_person_visible
                
                if visible_person:
                    # CASE 1: Face Recognized -> Switch immediately
                    last_person_time = time.time()
                    if current_agent_user != visible_person:
                        update_agent_for_user(visible_person, tools)
                
                elif person_detected:
                    # CASE 2: Person Detected but No Face (Looking down/back turned)
                    # KEEP current profile. Assume it's the same person.
                    last_person_time = time.time()
                    # Do not switch to Unknown here.
                        
                else:
                    # CASE 3: No Person Visible (Empty Room)
                    # Wait 15 seconds before switching to Unknown
                    if current_agent_user != "Unknown":
                        if time.time() - last_person_time > 15.0:
                             logging.info("⏳ Profile timeout: Switching to Unknown")
                             update_agent_for_user("Unknown", tools)

                try:
                    if not conversation_mode:
                        logging.info("🎤 Listening for wake word...")
                        # Tune sensitivity - phrase_time_limit helps avoid stuck listening
                        audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                        
                        logging.info("🎧 Processing audio...")
                        transcript = recognizer.recognize_google(audio)
                        logging.info(f"🗣 Heard: '{transcript}'")

                        if TRIGGER_WORD.lower() in transcript.lower():
                            logging.info(f"✅ Wake word '{TRIGGER_WORD}' detected!")
                            speak_text("Yes sir?")
                            conversation_mode = True
                            last_interaction_time = time.time()
                        else:
                            logging.debug(f"Captured '{transcript}', waiting for '{TRIGGER_WORD}'...")

                    else:
                        logging.info("🎤 Listening for next command...")
                        audio = recognizer.listen(source, timeout=10)
                        command = recognizer.recognize_google(audio)
                        logging.info(f"📥 Command: {command}")

                        # 1. Check for exit commands
                        exit_commands = ["no", "no thanks", "no thank you", "stop", "exit", "quit", "nothing", "that's all", "bye", "goodbye"]
                        # Check if command matches exactly or contains exit phrase at end
                        cmd_lower = command.lower().strip(".,!?")
                        if cmd_lower in exit_commands or any(cmd_lower.endswith(exit_cmd) for exit_cmd in exit_commands):
                            logging.info("🛑 Exit command received.")
                            speak_text("Alright, goodbye!")
                            conversation_mode = False
                            continue

                        # 2. Check for simple math (local processing)
                        math_result = solve_simple_math(command)
                        if math_result:
                            logging.info(f"🧮 Solved math locally: {math_result}")
                            print("Echo:", math_result)
                            speak_text(math_result)
                            last_interaction_time = time.time()
                            continue

                        # 3. Check for local vision intent (FAST PATH)
                        vision_result = check_vision_intent(command, shared_state)
                        if vision_result:
                            logging.info(f"👁️ Local vision result: {vision_result}")
                            print("Echo:", vision_result)
                            speak_text(vision_result)
                            last_interaction_time = time.time()
                            continue

                        # 4. Check for learning intent
                        learning_result = check_learning_intent(command, vision)
                        if learning_result:
                            logging.info(f"🎓 Learning result: {learning_result}")
                            print("Echo:", learning_result)
                            speak_text(learning_result)
                            last_interaction_time = time.time()
                            continue

                        # 5. Check for "Who am I" intent (Fast Path)
                        who_am_i_result = check_who_am_i_intent(command, face_tools)
                        if who_am_i_result:
                            logging.info(f"👤 Who am I result: {who_am_i_result}")
                            print("Echo:", who_am_i_result)
                            speak_text(who_am_i_result)
                            last_interaction_time = time.time()
                            continue

                        # 6. Check for face registration intent
                        identity_result = check_identity_intent(command, face_tools)
                        if identity_result:
                            logging.info(f"👤 Identity result: {identity_result}")
                            print("Echo:", identity_result)
                            speak_text(identity_result)
                            last_interaction_time = time.time()
                            continue

                        # 6. Check for local memory intent
                        local_memory_result = check_local_intent(command)
                        if local_memory_result:
                            logging.info(f"🧠 Local memory result: {local_memory_result}")
                            print("Echo:", local_memory_result)
                            speak_text(local_memory_result)
                            last_interaction_time = time.time()
                            continue

                        logging.info("🤖 Sending command to agent...")
                        
                        # Inject Vision Context
                        vision_context = shared_state.get_vision_summary()
                        full_input = f"Visual Context: {vision_context}\n\nUser Command: {command}"
                        logging.info(f"👀 Vision Context: {vision_context}")
                        
                        response = executor.invoke({"input": full_input})
                        content = response["output"]
                        logging.info(f"✅ Agent responded: {content}")

                        print("Echo:", content)
                        speak_text(content)
                        last_interaction_time = time.time()

                        if time.time() - last_interaction_time > CONVERSATION_TIMEOUT:
                            logging.info("⌛ Timeout: Returning to wake word mode.")
                            conversation_mode = False

                except sr.WaitTimeoutError:
                    logging.warning("⚠️ Timeout waiting for audio.")
                    if (
                        conversation_mode
                        and time.time() - last_interaction_time > CONVERSATION_TIMEOUT
                    ):
                        logging.info(
                            "⌛ No input in conversation mode. Returning to wake word mode."
                        )
                        conversation_mode = False
                except sr.UnknownValueError:
                    logging.warning("⚠️ Could not understand audio.")
                except sr.RequestError as e:
                    logging.error(f"❌ Connection error: {e}")
                    speak_text("I'm having trouble connecting to the internet. Please check my connection.")
                    time.sleep(5) # Cooldown to avoid flooding
                except Exception as e:
                    logging.error(f"❌ Error during recognition or tool call: {e}")
                    time.sleep(1)

    except Exception as e:
        logging.critical(f"❌ Critical error in main loop: {e}")


if __name__ == "__main__":
    write()
