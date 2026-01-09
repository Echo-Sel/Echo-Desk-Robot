import threading
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    timestamp: float

@dataclass
class VisionState:
    current_objects: List[DetectedObject] = field(default_factory=list)
    last_update_time: float = 0.0
    is_active: bool = False
    
class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._vision_state = VisionState()
        self._person_name: Optional[str] = None # Name of identified person

    def update_vision(self, objects: List[Dict[str, Any]]):
        with self._lock:
            detected_objects = []
            for obj in objects:
                detected_objects.append(DetectedObject(
                    label=obj.get('label', 'unknown'),
                    confidence=obj.get('confidence', 0.0),
                    bbox=obj.get('bbox', []),
                    timestamp=time.time()
                ))
            self._vision_state.current_objects = detected_objects
            self._vision_state.last_update_time = time.time()
            self._vision_state.is_active = True

    def update_person(self, name: Optional[str]):
        """Updates the identified person."""
        with self._lock:
            self._person_name = name

    @property
    def person_name(self) -> Optional[str]:
        with self._lock:
            return self._person_name

    @property
    def is_person_visible(self) -> bool:
        """Checks if a person object is currently detected by YOLO."""
        with self._lock:
            if not self._vision_state.is_active:
                return False
            for obj in self._vision_state.current_objects:
                if obj.label == "person":
                    return True
            return False
            
    def get_vision_summary(self) -> str:
        with self._lock:
            if not self._vision_state.is_active or (time.time() - self._vision_state.last_update_time > 5.0):
                return "Vision system is not active or hasn't updated recently."
            
            if not self._vision_state.current_objects:
                return "I don't see any specific objects right now."
            
            # Simple summary
            counts = {}
            for obj in self._vision_state.current_objects:
                counts[obj.label] = counts.get(obj.label, 0) + 1
            
            
            summary_parts = []
            for label, count in counts.items():
                # If we see a person and know their name, use it!
                if label == "person" and self._person_name:
                    if count == 1:
                        summary_parts.append(f"{self._person_name}")
                    else:
                        # Multiple people, one is recognized
                        summary_parts.append(f"{self._person_name} and {count-1} other{'s' if count-1 > 1 else ''}")
                else:
                    if count == 1:
                        summary_parts.append(f"a {label}")
                    else:
                        summary_parts.append(f"{count} {label}s")
            
            objects_str = ", ".join(summary_parts)
            
            # Personality templates (Cool/Casual Vibe)
            templates = [
                f"Yo, I see {objects_str} right there.",
                f"Spying {objects_str}.",
                f"Check it out, {objects_str} in the frame.",
                f"I'm lookin' at {objects_str}.",
                f"Just {objects_str} hanging out."
            ]
            
            return random.choice(templates)

    def get_raw_vision_data(self) -> List[Dict[str, Any]]:
         with self._lock:
            return [
                {
                    "label": obj.label,
                    "confidence": obj.confidence
                }
                for obj in self._vision_state.current_objects
            ]
