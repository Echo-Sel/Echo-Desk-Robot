from langchain.tools import tool
from typing import List
import logging

class FaceRecognitionTools:
    def __init__(self, vision_system):
        self.vision_system = vision_system

    def register_face(self, name: str) -> str:
        """
        Learns the face of the person currently looking at the camera.
        Use this when a user says "Remember me" or "Learn my face".
        """
        # Triggers the training mode in the background thread
        return self.vision_system.start_registration(name)

    def recognize_face(self, query: str = "") -> str:
        """
        Identifies the person currently in front of the camera.
        Use this when asked "Who am I?" or "Do you know me?".
        """
        # The vision system updates SharedState continuously. 
        # We just check the latest identified person.
        person = self.vision_system.shared_state.person_name
        
        if person:
            return f"I recognize {person}!"
        else:
            return "I don't recognize anyone right now. (Make sure you're looking at the camera)"

    def list_known_people(self, query: str = "") -> str:
        """
        Lists all people whose faces I have learned.
        """
        names = list(self.vision_system.known_names.values())
        if not names:
            return "I don't know anyone yet."
        return f"I know the following people: {', '.join(names)}"

    def forget_person(self, name: str) -> str:
        """
        Forgets a person's face. 
        Use this when asked to delete someone from memory.
        """
        return self.vision_system.forgetting_person(name)
