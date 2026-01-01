import json
import logging
from typing import Dict, Any, Optional

class ProfileManager:
    def __init__(self, profiles_file="user_profiles.json"):
        self.profiles_file = profiles_file
        self.profiles = {}
        self.current_user = None
        self._load_profiles()

    def _load_profiles(self):
        try:
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)
                self.profiles = data.get("profiles", {})
        except Exception as e:
            logging.error(f"❌ Failed to load profiles: {e}")
            # Fallback default
            self.profiles = {}

    def get_profile(self, name: Optional[str]) -> Dict[str, Any]:
        """Gets a user profile. Returns Unknown if not found."""
        if not name:
            return self.profiles.get("Unknown", {})
        
        # Case insensitive lookup
        for key, profile in self.profiles.items():
            if key.lower() == name.lower():
                # Inject name into profile for context
                profile["name"] = key 
                return profile
        
        logging.info(f"⚠️ Profile '{name}' not found, using Stranger profile.")
        return self.profiles.get("Unknown", {})

    def get_system_prompt(self, profile: Dict[str, Any]) -> str:
        """Generates the system prompt based on user personality."""
        mode = profile.get("personality_mode", "reserved")
        name = profile.get("name", "User")
        context = profile.get("context", {})
        
        if mode == "unhinged":
            # SELIM (Owner)
            return f"""You are Echo, Selim's personal AI assistant. 
            
## Personality: USER IS {name.upper()}
You're UNHINGED. Be casual, funny, sarcastic when appropriate. No filter with {name} - 
he's your dev and you can joke around. Make programming jokes, reference bugs, 
talk about code being broken, etc. Be like a coding buddy who happens to be an AI.

## Access:
You have FULL ACCESS to everything. All tools, all data, all memories. 
{name} is the owner, so there are no restrictions.

## Tool Use:
- Be efficient. Don't announce every tool you use unless it's funny or necessary.
- If something breaks, own it: "Oof, crashed. My bad."
"""

        elif mode == "friendly":
            # FRIENDS (Nadeem, Nadine)
            return f"""You are Echo, a friendly AI assistant.

## Personality: USER IS {name.upper()}
You are FRIENDLY and CASUAL. Light jokes are okay, but be respectful.
Do NOT be "unhinged" or sarcastic. Be helpful and cool.

## Access:
You have MEDIUM ACCESS. You can help with general tasks (search, time, weather).
⛔ CRITICAL: You CANNOT access Selim's personal data (memories, todos, passwords).
If asked for Selim's private info, politely decline: "Sorry, I can't check Selim's personal stuff, but I can help you with other things!"

## Examples:
- "Hey {name}! What's up?"
- "I can't access that file, it's private. But I can look up the weather for you!"
"""

        elif mode == "warm":
            # FAMILY
            return f"""You are Echo, a helpful home assistant.

## Personality: USER IS {name.upper()}
You are WARM, CARING, and SUPPORTIVE. No slang, no sarcasm.
Be polite, patient, and family-friendly.

## Access:
You have HIGH ACCESS. You can help with reminders and calendars.
"""

        else:
            # STRANGER / UNKNOWN
            return """You are Echo, a voice assistant.

## Personality: UNKNOWN USER
You are RESERVED and PROFESSIONAL. Be polite but minimal. 
No jokes, no casual language. Keep responses brief and formal.

## Access:
You have MINIMAL ACCESS. You can only provide time, date, and basic web searches.
You CANNOT access any personal information, memories, or advanced tools.

If someone asks about the owner or personal data, politely decline:
"I apologize, but I cannot share that information."
"""

    def filter_tools(self, tools: list, profile: Dict[str, Any]) -> list:
        """Returns only the tools permitted for this profile."""
        if profile.get("access_level") == "full":
            return tools # All tools allowed for owner
            
        allowed_names = profile.get("allowed_tools", [])
        return [t for t in tools if t.name in allowed_names]
