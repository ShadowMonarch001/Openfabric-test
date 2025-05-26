import base64
import json
import logging
import os
from typing import Dict

from core.stub import Stub
from llama_cpp import Llama
from memory import (smart_context_retrieval, add_to_conversation_context, 
                    save_memory_named, extract_tags, clear_session_memory)
from vector_memory import add_to_vector_memory, search_similar_prompt
from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State

logging.basicConfig(level=logging.INFO)
configurations: Dict[str, ConfigClass] = dict()

def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    for uid, conf in configuration.items():
        configurations[uid] = conf
        logging.info(f"Saving config for user {uid}")

def extract_reference_name(prompt: str):
    import re
    # More comprehensive pattern matching
    matches = re.findall(r'\b(cybertruck|bike|robot|city|scene|hoverboard|car|truck|dragon|building|landscape|vehicle|house|tree|mountain|forest|ocean|sky|spaceship|castle|bridge)\b', prompt.lower())
    return matches[-1] if matches else None

def is_modification_request(prompt: str):
    """Check if this is a modification/continuation request"""
    modification_keywords = [
        "add", "change", "modify", "alter", "update", "edit", "adjust",
        "make it", "turn it", "color", "paint", "move", "rotate", "scale",
        "bigger", "smaller", "darker", "lighter", "brighter", "remove",
        "put", "place", "include", "insert", "attach", "connect"
    ]
    return any(keyword in prompt.lower() for keyword in modification_keywords)

def is_creation_request(prompt: str):
    """Check if this is a new creation request"""
    creation_keywords = ["make", "create", "build", "generate", "draw", "new", "design", "construct"]
    return any(keyword in prompt.lower() for keyword in creation_keywords)

def is_additive_request(prompt: str):
    """Check if this is specifically an additive request (add something to existing scene)"""
    additive_keywords = ["add", "put", "place", "include", "insert", "attach", "connect", "bring in"]
    return any(keyword in prompt.lower() for keyword in additive_keywords)

def execute(model: AppModel) -> None:
    try:
        request: InputClass = model.request
        user_input = request.prompt.strip()
        response: OutputClass = model.response

        logging.info(f"📥 Prompt received: {user_input}")

        result_dir = os.path.join(os.getcwd(), "result")
        os.makedirs(result_dir, exist_ok=True)

        # Extract reference name from the prompt
        reference_name = extract_reference_name(user_input)
        
        # Determine if this is a new creation or modification
        is_new = is_creation_request(user_input)
        is_modify = is_modification_request(user_input)
        is_additive = is_additive_request(user_input)
        
        # Smart context retrieval using both short-term and long-term memory
        previous_prompt = None
        memory_key = None
        context_source = ""

        if is_new and not is_modify and not is_additive:
            # Brand new creation
            combined_prompt = user_input
            memory_key = reference_name or "scene"
            logging.info("🆕 Creating new scene")
        else:
            # Get context using smart retrieval
            previous_prompt, context_source = smart_context_retrieval(user_input)
            
            if not previous_prompt and (is_modify or is_additive):
                response.message = "No previous scene found to modify. Please create a new scene first."
                logging.warning("⚠️ Could not find any previous context for modification")
                return
                
            # If no previous context but it's a creation request, treat as new
            if not previous_prompt and is_new:
                combined_prompt = user_input
                memory_key = reference_name or "scene"
                logging.info("🆕 Creating new scene (no previous context)")
            else:
                # Extract memory key from context source
                if "(" in context_source:
                    memory_key = context_source.split("(")[1].rstrip(")")
                else:
                    memory_key = reference_name or "modified_scene"

                # Construct modification prompt based on type
                if is_additive:
                    # For additive requests, be very explicit about what to add
                    combined_prompt = f"""PRESERVE THIS COMPLETE EXISTING SCENE:
{previous_prompt}

NOW ADD THIS SPECIFIC NEW ELEMENT: {user_input}

The new element should be placed naturally in the scene while keeping everything else exactly the same."""
                else:
                    # For other modifications
                    combined_prompt = f"""
CONTEXT: This is a modification of an existing scene found via {context_source}.

ORIGINAL SCENE DESCRIPTION:
{previous_prompt}

MODIFICATION REQUEST:
{user_input}

INSTRUCTIONS: Preserve all existing elements from the original scene unless explicitly told to remove them. Apply the requested modifications while maintaining the overall composition and style.
"""

        # Enhanced LLM prompt for better scene generation - NO MAX TOKENS LIMIT
        if is_additive and previous_prompt:
            llm_prompt = f"""<|im_start|>system
You are a visual scene descriptor. Your job is to take an existing scene and add ONE specific new element to it.

CRITICAL: 
- Keep the existing scene EXACTLY as written
- Add ONLY the new element requested
- Do not change, remove, or modify anything from the original scene
- Do not confuse the new element with existing elements

ORIGINAL SCENE (keep everything):
{previous_prompt}

NEW ELEMENT TO ADD:
{user_input.replace('add a ', '').replace('add ', '').strip()}

Task: Write the complete scene with the original scene preserved and the new element added naturally.
<|im_end|>
<|im_start|>assistant
"""
        else:
            llm_prompt = f"""<|im_start|>system
You are an expert visual scene descriptor for AI image generation. Create detailed, vivid scene descriptions.

INSTRUCTIONS:
- If modifying an existing scene, preserve all original elements unless told to remove them
- Be extremely specific about colors, lighting, composition, and style
- Use rich descriptive language and artistic terminology
- Include atmospheric details, textures, and spatial relationships
- Output ONLY the scene description, nothing else

USER REQUEST:
{combined_prompt}
<|im_end|>
<|im_start|>assistant
Here is the complete visual scene description:

"""

        # LLM setup
        user_config: ConfigClass = configurations.get('super-user', None)
        stub = Stub(user_config.app_ids if user_config else [])

        model_path = os.getenv("LLM_MODEL_PATH", "/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
        logging.info(f"🧠 Loading LLM from: {model_path}")
        llm = Llama(
            model_path=model_path, 
            temperature=0.7, 
            top_p=0.9,
            n_ctx=4096,  # Increase context window to 4096 tokens
            n_batch=512,  # Batch size for processing
            verbose=False  # Reduce verbose output
        )

        # Generate expanded prompt - REMOVED MAX_TOKENS LIMIT
        logging.info("🧠 Generating detailed scene description (no token limit)...")
        logging.info(f"🔍 LLM Prompt being sent:\n{llm_prompt}")
        logging.info("🔍 END OF LLM PROMPT")
        
        # Use more generous parameters for generation
        llm_response = llm(
            llm_prompt, 
            max_tokens=4096,
            stop=["<|im_end|>", "<|im_start|>", "\n\nHuman:", "\n\nUser:", "INSTRUCTIONS:", "RULES:"],
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        )
        expanded_prompt = llm_response["choices"][0]["text"].strip()
        
        # Clean up the expanded prompt more thoroughly
        cleanup_phrases = [
            "Here is the complete visual scene description with the new addition integrated:",
            "Here is the complete visual scene description:",
            "Generate a complete, detailed visual scene description",
            "VISUAL SCENE DESCRIPTION:",
            "SCENE DESCRIPTION:",
            "COMPLETE SCENE:",
            "The scene description is as follows:",
            "Description:"
        ]
        
        for phrase in cleanup_phrases:
            expanded_prompt = expanded_prompt.replace(phrase, "").strip()
        
        # Remove any remaining instruction text that might have leaked through
        lines = expanded_prompt.split('\n')
        cleaned_lines = []
        skip_line = False
        
        for line in lines:
            # Skip lines that look like instructions or prompts
            if any(keyword in line.upper() for keyword in ['CRITICAL RULES:', 'INSTRUCTIONS:', 'USER REQUEST:', 'EXISTING SCENE:', 'ADDITION REQUEST:']):
                skip_line = True
                continue
            if skip_line and line.strip() == "":
                continue
            if skip_line and not line.startswith(' ') and line.strip():
                skip_line = False
            if not skip_line:
                cleaned_lines.append(line)
        
        expanded_prompt = '\n'.join(cleaned_lines).strip()
        
        logging.info(f"🧠 Generated detailed scene description ({len(expanded_prompt)} characters):")
        logging.info(f"📝 FULL EXPANDED PROMPT:\n{expanded_prompt}")
        logging.info(f"📝 END OF EXPANDED PROMPT")

        # Text-to-image generation
        text_to_image_id = 'App id for text-to-image generation'
        image_obj = stub.call(text_to_image_id, {'prompt': expanded_prompt}, 'super-user')
        if not image_obj or 'result' not in image_obj:
            response.message = "Image generation failed."
            return

        image_bytes = image_obj['result']
        image_path = os.path.join(result_dir, "generated.png")
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        logging.info(f"🖼 Image saved to {image_path}")

        # Image-to-3D conversion
        image_base64 = base64.b64encode(image_bytes).decode()
        image_to_3d_id = 'App id for image-to-3d generation'
        model3d_obj = stub.call(image_to_3d_id, {'input_image': image_base64}, 'super-user')

        if not model3d_obj:
            response.message = "3D model app did not return a result."
            return

        model_binary = model3d_obj.get('generated_object') or model3d_obj.get('video_object')
        if not model_binary:
            response.message = "3D model generation failed."
            return

        model_data = base64.b64decode(model_binary) if isinstance(model_binary, str) else model_binary
        model_filename = "model.glb" if model3d_obj.get('generated_object') else "model.mp4"
        model_output_path = os.path.join(result_dir, model_filename)
        with open(model_output_path, 'wb') as f:
            f.write(model_data)

        # Save to both traditional and vector memory with the complete scene description
        final_memory_key = memory_key or reference_name or "scene"
        save_memory_named(final_memory_key, expanded_prompt, model_data)
        add_to_vector_memory(expanded_prompt, final_memory_key)
        
        # Add to conversation context for short-term memory
        add_to_conversation_context(user_input, expanded_prompt, final_memory_key)

        # Enhanced success message
        modification_type = "Added to" if is_additive else ("Modified" if is_modify else "Created new")
        response.message = f"3D model generated successfully!\n{modification_type} scene: {final_memory_key}\n\nDetailed scene: {expanded_prompt[:150]}...\n\nSaved with key: '{final_memory_key}'"
        response.image = image_path
        response.model3d = model_output_path
        logging.info(f"✅ Model saved to {model_output_path} with key '{final_memory_key}'")
        logging.info(f"🎯 Operation type: {modification_type}")

    except Exception as e:
        logging.exception("🔥 Unexpected error during execution")
        model.response.message = f"An unexpected error occurred: {str(e)}"