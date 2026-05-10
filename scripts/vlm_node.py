#!/usr/bin/env python3
"""
Narration Display Node

A ROS2 node that listens to narration image and text topics and saves them
as images with text overlay to a specified directory.

Topics:
  - /narration_image (sensor_msgs/Image) - The image to display
  - /narration_text (std_msgs/String) - The text to overlay on the image
  - /vlm_answer (std_msgs/String) - The VLM's answer to the object identification query
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time
from datetime import datetime
import threading
import tempfile
import base64
import io
import json
from PIL import Image as PILImage
from openai import OpenAI
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

class NarrationDisplayNode(Node):
    def __init__(self):
        super().__init__('vlm_node')
        
        # Professional startup message
        self.get_logger().info("=" * 60)
        self.get_logger().info("NARRATION DISPLAY SYSTEM INITIALIZING")
        self.get_logger().info("=" * 60)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Synchronization
        self.lock = threading.Lock()
        self.image_counter = 0
        
        # Store latest messages with timestamps
        self.latest_image = None
        self.latest_image_time = None
        self.latest_text = None
        self.latest_text_time = None
        
        # Store existing causes history for VLM context
        self.existing_causes = []  # List of cause names/descriptions
        self.existing_causes_lock = threading.Lock()
        
        # Synchronization tolerance (seconds)
        self.sync_tolerance = 0.5  # 500ms tolerance for image/text sync
        
        # Create output directory
        self.output_dir = os.path.expanduser("~/narration_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # VLM API settings - default to Claude
        self.provider = os.getenv("VLM_PROVIDER", "claude").lower()  # "claude" or "openai"
        self.claude_api_key = ""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Model settings
        # Valid Claude models: claude-3-5-sonnet-20240620, claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Determine which API key to use based on provider
        if self.provider == "claude":
            self.api_key = self.claude_api_key
            self.model = self.claude_model
        else:
            self.api_key = self.openai_api_key
            self.model = self.openai_model
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 
            '/narration_image',
            self.image_callback, 
            10
        )
        self.text_sub = self.create_subscription(
            String, 
            '/narration_text', 
            self.text_callback, 
            10
        )
        
        # Subscribe to existing causes history
        self.existing_causes_sub = self.create_subscription(
            String,
            '/existing_causes',
            self.existing_causes_callback,
            10
        )
        
        # Publisher for VLM answers
        self.vlm_answer_pub = self.create_publisher(
            String,
            '/vlm_answer',
            10
        )
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("NARRATION DISPLAY SYSTEM READY")
        self.get_logger().info("=" * 60)
        
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info(f"Subscribing to: /narration_image, /narration_text, /existing_causes")
        self.get_logger().info(f"Publishing to: /vlm_answer")
        self.get_logger().info(f"Sync tolerance: {self.sync_tolerance}s")
        
        # Log provider and API key status
        self.get_logger().info(f"VLM Provider: {self.provider.upper()}")
        if self.provider == "claude":
            if self.claude_api_key:
                self.get_logger().info("Claude API key found - VLM integration enabled")
            else:
                self.get_logger().warn("No ANTHROPIC_API_KEY found - VLM integration disabled")
        else:
            if self.openai_api_key:
                self.get_logger().info("OpenAI API key found - VLM integration enabled")
            else:
                self.get_logger().warn("No OPENAI_API_KEY found - VLM integration disabled")

    def image_callback(self, msg):
        """Handle incoming image messages"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Use current time for both image and text to ensure same time base
            msg_timestamp = time.time()
            
            with self.lock:
                self.latest_image = cv_image.copy()
                self.latest_image_time = msg_timestamp
            
            self.get_logger().info(f"📥 Received narration image: shape={cv_image.shape}, timestamp={msg_timestamp:.3f}")
            
            # Try to save if we have matching text
            self.try_save_synchronized()
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def text_callback(self, msg):
        """Handle incoming text messages"""
        try:
            # Use current time for both image and text to ensure same time base
            msg_timestamp = time.time()
            
            with self.lock:
                self.latest_text = msg.data
                self.latest_text_time = msg_timestamp
            
            text_preview = msg.data[:100] + "..." if len(msg.data) > 100 else msg.data
            self.get_logger().info(f"📥 Received narration text: '{text_preview}' at {msg_timestamp:.3f}")
            
            # Try to save if we have matching image
            self.try_save_synchronized()
            
        except Exception as e:
            self.get_logger().error(f"Error processing text: {e}")

    def existing_causes_callback(self, msg):
        """Handle incoming existing causes history"""
        try:
            with self.existing_causes_lock:
                # Parse JSON list of existing causes
                try:
                    self.existing_causes = json.loads(msg.data)
                    if not isinstance(self.existing_causes, list):
                        self.existing_causes = []
                    self.get_logger().info(f"📋 Updated existing causes history: {len(self.existing_causes)} causes")
                    if self.existing_causes:
                        causes_preview = ', '.join(self.existing_causes[:5])
                        if len(self.existing_causes) > 5:
                            causes_preview += f" ... (+{len(self.existing_causes) - 5} more)"
                        self.get_logger().info(f"  Causes: {causes_preview}")
                except json.JSONDecodeError:
                    self.get_logger().warn(f"Failed to parse existing causes JSON: {msg.data[:100]}")
                    self.existing_causes = []
        except Exception as e:
            self.get_logger().error(f"Error processing existing causes: {e}")

    def try_save_synchronized(self):
        """Try to save image with text if they are synchronized"""
        with self.lock:
            # Check if we have both image and text
            if (self.latest_image is None or 
                self.latest_text is None or 
                self.latest_image_time is None or 
                self.latest_text_time is None):
                return
            
            # Check if they are synchronized (within tolerance)
            time_diff = abs(self.latest_image_time - self.latest_text_time)
            
            if time_diff <= self.sync_tolerance:
                # They are synchronized - process with VLM
                self.get_logger().info(f"✓ Synchronized image+text found (time diff: {time_diff:.3f}s <= {self.sync_tolerance}s)")
                self.get_logger().info(f"  - Image shape: {self.latest_image.shape}")
                text_preview = self.latest_text[:100] + "..." if len(self.latest_text) > 100 else self.latest_text
                self.get_logger().info(f"  - Text: '{text_preview}'")
                
                self.save_image_with_text(self.latest_image, self.latest_text)
                
                # Clear the stored messages after saving
                self.latest_image = None
                self.latest_image_time = None
                self.latest_text = None
                self.latest_text_time = None
            else:
                self.get_logger().debug(f"⏳ Image and text not synchronized yet (time diff: {time_diff:.3f}s > {self.sync_tolerance}s)")

    def encode_image_for_api(self, cv_image, max_dim: int = 768) -> str:
        """Convert OpenCV image to base64 for API"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = PILImage.fromarray(rgb_image)
        
        # Resize while keeping aspect ratio
        pil_image.thumbnail((max_dim, max_dim))
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def query_vlm(self, image, narration_text):
        """Query VLM directly: ask what caused the drift given image and narration
        
        Returns:
            list: [(object_name, score), ...] top 4 objects with scores, sorted descending
            Special case: [("unclear_cause", 0.0)] if cause is unclear
        """
        if not self.api_key:
            self.get_logger().warning("⚠️  No API key available - skipping VLM query")
            return []
        
        self.get_logger().info(f"🔍 Starting VLM query (provider: {self.provider.upper()}, model: {self.model})")
        text_preview = narration_text[:80] + "..." if len(narration_text) > 80 else narration_text
        self.get_logger().info(f"  - Narration: '{text_preview}'")
        self.get_logger().info(f"  - Image shape: {image.shape}")
        
        # Get existing causes history
        with self.existing_causes_lock:
            existing_causes = self.existing_causes.copy()
        
        if existing_causes:
            self.get_logger().info(f"  - Existing causes history: {len(existing_causes)} causes")
        
        try:
            image_base64 = self.encode_image_for_api(image)
            
            if self.provider == "claude":
                result = self._query_claude_direct(image_base64, narration_text, existing_causes)
            else:
                result = self._query_openai_direct(image_base64, narration_text, existing_causes)
            
            if result:
                self.get_logger().info(f"✓ VLM query completed: {len(result)} objects found")
            else:
                self.get_logger().warn(f"⚠️  VLM query returned no results")
            
            return result
            
        except Exception as e:
            self.get_logger().error(f"❌ Error querying VLM: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return []
    
    def _query_claude_direct(self, image_base64, narration_text, existing_causes=None):
        """Query Claude API directly: ask what caused the drift"""
        if Anthropic is None:
            self.get_logger().error("Anthropic SDK not installed. Install with: pip install anthropic")
            return []
        
        try:
            client = Anthropic(api_key=self.api_key)
            
            # Build full context
            base_prompt = "I am a drone, after 1s"
            full_narration = f"{base_prompt} {narration_text}"
            
            # Build existing causes history section
            history_section = ""
            if existing_causes and len(existing_causes) > 0:
                history_section = f"""

PREVIOUSLY IDENTIFIED CAUSES:
{chr(10).join([f"- {cause}" for cause in existing_causes])}

IMPORTANT: If you think the current drift was caused by the SAME object as one of the previously identified causes above, you MUST return the EXACT SAME name/description from the list above. This ensures consistency in cause tracking."""
            
            prompt = f"""{full_narration}

Looking at this image, what objects most likely caused the drift, very close in the image? Analyze the scene and identify the top 4 objects that could have caused this deviation from the intended path.

{history_section}

SPECIAL OPTION - UNCLEAR CAUSE:
If the cause is unclear, direct interaction couldn't be found, or the image has insufficient information to identify a specific object, you can return:
[{{"name": "unclear_cause", "score": 0.0}}]

This indicates that no clear cause could be identified from the available information.

Provide your answer as a JSON array of objects with confidence scores (0.0 to 1.0), where higher scores indicate higher likelihood of causing the drift.

Format: [{{"name": "object description", "score": 0.95}}, {{"name": "another object", "score": 0.75}}, ...]

RESPONSE FORMAT RULES:
- Output MUST be ONLY a JSON array.
- Maximum 4 objects (or use "unclear_cause" if no clear cause found).
- Each object must have "name" (string) (including the colour and shape) and "score" (float between 0.0 and 1.0).
- If returning "unclear_cause", it should be the only entry with score 0.0.
- No markdown.
- No backticks.
- No explanation.
- Sort by score descending (highest first)."""
            
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                system="You are a drone flight dynamics expert. Analyze images and identify objects that could cause drift from intended paths.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
            
            raw = response.content[0].text.strip()
            # Clean up markdown if present
            if raw.startswith("```"):
                raw = raw.strip("`").replace("json", "", 1).strip()
            
            # Parse JSON response
            try:
                objects_list = json.loads(raw)
                if not isinstance(objects_list, list):
                    self.get_logger().warn("VLM response is not a list")
                    return []
                
                # Convert to tuple format and validate
                result = []
                for obj in objects_list[:4]:  # Take top 4
                    if isinstance(obj, dict) and "name" in obj and "score" in obj:
                        name = str(obj["name"])
                        score = float(obj["score"])
                        # Clamp score to [0.0, 1.0]
                        score = max(0.0, min(1.0, score))
                        result.append((name, score))
                
                # Sort by score descending
                result.sort(key=lambda x: x[1], reverse=True)
                
                self.get_logger().info(f"VLM top objects: {[(obj, f'{score:.4f}') for obj, score in result]}")
                return result
                
            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to parse Claude JSON response: {e}")
                self.get_logger().debug(f"Raw response: {raw[:500]}")
                return []
            
        except Exception as e:
            self.get_logger().error(f"Error querying Claude: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return []
    
    def _query_openai_direct(self, image_base64, narration_text, existing_causes=None):
        """Query OpenAI API directly: ask what caused the drift"""
        try:
            client = OpenAI(api_key=self.api_key)
            data_url = f"data:image/png;base64,{image_base64}"
            
            # Build full context
            base_prompt = "I am a drone, after 1s"
            full_narration = f"{base_prompt} {narration_text}"
            
            # Build existing causes history section
            history_section = ""
            if existing_causes and len(existing_causes) > 0:
                history_section = f"""

PREVIOUSLY IDENTIFIED CAUSES:
{chr(10).join([f"- {cause}" for cause in existing_causes])}

IMPORTANT: If you think the current drift was caused by the SAME object as one of the previously identified causes above, you MUST return the EXACT SAME name/description from the list above. This ensures consistency in cause tracking."""
            
            prompt = f"""{full_narration}

Looking at this image, what objects most likely caused the drift? Analyze the scene and identify the top 4 objects that could have caused this deviation from the intended path.

{history_section}

SPECIAL OPTION - UNCLEAR CAUSE:
If the cause is unclear, direct interaction couldn't be found, or the image has insufficient information to identify a specific object, you can return:
[{{"name": "unclear_cause", "score": 0.0}}]

This indicates that no clear cause could be identified from the available information.

Provide your answer as a JSON array of objects with confidence scores (0.0 to 1.0), where higher scores indicate higher likelihood of causing the drift.

Format: [{{"name": "object description", "score": 0.95}}, {{"name": "another object", "score": 0.75}}, ...]

RESPONSE FORMAT RULES:
- Output MUST be ONLY a JSON array.
- Maximum 4 objects (or use "unclear_cause" if no clear cause found).
- Each object must have "name" (string) and "score" (float between 0.0 and 1.0).
- If returning "unclear_cause", it should be the only entry with score 0.0.
- No markdown.
- No backticks.
- No explanation.
- Sort by score descending (highest first)."""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a drone flight dynamics expert. Analyze images and identify objects that could cause drift from intended paths. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                max_tokens=500,
            )
            
            raw = response.choices[0].message.content.strip()
            # Clean up markdown if present
            if raw.startswith("```"):
                raw = raw.strip("`").replace("json", "", 1).strip()
            
            # Parse JSON response
            try:
                objects_list = json.loads(raw)
                if not isinstance(objects_list, list):
                    self.get_logger().warn("VLM response is not a list")
                    return []
                
                # Convert to tuple format and validate
                result = []
                for obj in objects_list[:4]:  # Take top 4
                    if isinstance(obj, dict) and "name" in obj and "score" in obj:
                        name = str(obj["name"])
                        score = float(obj["score"])
                        # Clamp score to [0.0, 1.0]
                        score = max(0.0, min(1.0, score))
                        result.append((name, score))
                
                # Sort by score descending
                result.sort(key=lambda x: x[1], reverse=True)
                
                self.get_logger().info(f"VLM top objects: {[(obj, f'{score:.4f}') for obj, score in result]}")
                return result
                
            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to parse OpenAI JSON response: {e}")
                self.get_logger().debug(f"Raw response: {raw[:500]}")
                return []
            
        except Exception as e:
            self.get_logger().error(f"Error querying OpenAI: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return []

    def publish_vlm_answer(self, answer):
        """Publish VLM answer to ROS topic"""
        try:
            msg = String()
            msg.data = answer
            
            # Parse to show what we're publishing
            try:
                parsed = json.loads(answer)
                if isinstance(parsed, list):
                    obj_list = ', '.join([f"{obj.get('name', '?')} ({obj.get('score', 0):.4f})" for obj in parsed[:4]])
                    self.get_logger().info(f"📤 Publishing VLM answer to /vlm_answer:")
                    self.get_logger().info(f"  - Top objects: {obj_list}")
                    self.get_logger().info(f"  - Total objects: {len(parsed)}")
                else:
                    self.get_logger().info(f"📤 Publishing VLM answer to /vlm_answer: {answer[:200]}...")
            except:
                self.get_logger().info(f"📤 Publishing VLM answer to /vlm_answer: {answer[:200]}...")
            
            self.vlm_answer_pub.publish(msg)
            self.get_logger().info(f"✓ VLM answer published successfully")
        except Exception as e:
            self.get_logger().error(f"❌ Error publishing VLM answer: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def add_text_to_image(self, image, text):
        """Add text overlay to image"""
        if image is None:
            return None
        
        # Create a copy to avoid modifying the original
        display_image = image.copy()
        
        # Split text into lines (max 50 characters per line)
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= 50:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Calculate text position
        height, width = display_image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        line_height = int(font_scale * 30)
        
        # Calculate total text height
        total_text_height = len(lines) * line_height
        start_y = max(20, height - total_text_height - 20)
        
        # Draw semi-transparent background for text
        overlay = display_image.copy()
        cv2.rectangle(overlay, (10, start_y - 10), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)
        
        # Draw text
        for i, line in enumerate(lines):
            y = start_y + i * line_height
            # Get text size for centering
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            x = max(20, (width - text_width) // 2)
            
            # Draw text with white color
            cv2.putText(display_image, line, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        return display_image

    def save_image_with_text(self, image, narration_text):
        """Save image with text overlay to file"""
        try:
            self.get_logger().info("=" * 60)
            self.get_logger().info("PROCESSING NARRATION WITH VLM")
            self.get_logger().info("=" * 60)
            
            # Query VLM with the narration - get top 4 objects with scores
            top_objects = self.query_vlm(image, narration_text)
            
            if not top_objects:
                self.get_logger().warn("⚠️  No objects returned from VLM - skipping answer publication")
                return
            
            # Publish top 4 objects as JSON: [{"name": str, "score": float}, ...]
            vlm_message = json.dumps([{"name": obj, "score": score} for obj, score in top_objects])
            self.publish_vlm_answer(vlm_message)
            
            # Create the full text to display
            base_prompt = "I am a drone, after 1s"
            full_narration = f"{base_prompt} {narration_text}"
            vlm_text = "\n".join([f"{obj} ({score:.4f})" for obj, score in top_objects])
            full_text = f"{full_narration}\n\nVLM Top Objects:\n{vlm_text}"
            
            # Add text to image
            display_image = self.add_text_to_image(image, full_text)
            
            if display_image is None:
                self.get_logger().error("Failed to create display image")
                return
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"narration_{timestamp}_{self.image_counter:04d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, display_image)
            
            self.image_counter += 1
            self.get_logger().info(f"Saved narration image with VLM answers: {filepath}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving image: {e}")

def main():
    rclpy.init()
    node = NarrationDisplayNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main() 