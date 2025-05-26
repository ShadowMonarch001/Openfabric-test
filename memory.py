import base64
import json
import os
import re
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

# Memory file paths
MEMORY_PATH = 'memory.json'
SESSION_MEMORY_PATH = 'session_memory.json'
CONVERSATION_CONTEXT_PATH = 'conversation_context.json'

class MemoryManager:
    def __init__(self):
        self.session_context = deque(maxlen=10)  # Last 10 interactions in current session
        self.conversation_buffer = []  # Current conversation thread
        self.load_session_state()
    
    def load_session_state(self):
        """Load current session state"""
        if os.path.exists(SESSION_MEMORY_PATH):
            try:
                with open(SESSION_MEMORY_PATH, 'r') as f:
                    data = json.load(f)
                    self.session_context = deque(data.get('session_context', []), maxlen=10)
                    self.conversation_buffer = data.get('conversation_buffer', [])
            except:
                pass
    
    def save_session_state(self):
        """Save current session state"""
        data = {
            'session_context': list(self.session_context),
            'conversation_buffer': self.conversation_buffer,
            'timestamp': time.time()
        }
        with open(SESSION_MEMORY_PATH, 'w') as f:
            json.dump(data, f, indent=2)

# Initialize global memory manager
memory_manager = MemoryManager()

def load_memory(path=MEMORY_PATH) -> Dict:
    """Load long-term memory from JSON file"""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load memory from {path}, starting fresh")
            return {}
    return {}

def extract_tags(prompt: str) -> List[str]:
    """Extract relevant tags from prompt for better searchability"""
    
    # Predefined important tags (your original ones + more)
    predefined_patterns = [
        r'\b(cybertruck|tesla|truck|vehicle|car|bike|motorcycle|hoverboard)\b',
        r'\b(robot|android|cyborg|mech|drone)\b', 
        r'\b(city|urban|downtown|skyline|street|building|skyscraper)\b',
        r'\b(scene|landscape|environment|background|setting)\b',
        r'\b(neon|futuristic|sci-fi|cyberpunk|modern|industrial)\b',
        r'\b(dragon|creature|animal|beast|monster|alien)\b',
        r'\b(red|blue|green|yellow|purple|orange|black|white|silver|gold|pink|cyan|magenta)\b',
        r'\b(bright|dark|glowing|metallic|shiny|matte|transparent|opaque)\b',
        r'\b(house|home|castle|tower|bridge|road|path|forest|mountain|ocean|lake|desert)\b',
        r'\b(flying|floating|moving|static|rotating|spinning|glowing|burning)\b'
    ]
    
    tags = []
    prompt_lower = prompt.lower()
    
    # Extract predefined patterns first
    for pattern in predefined_patterns:
        matches = re.findall(pattern, prompt_lower)
        tags.extend(matches)
    
    # Try advanced NLP extraction
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tag import pos_tag
        from nltk.tokenize import word_tokenize

        # Download required NLTK data (only first time)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Use NLP to extract additional nouns and adjectives
        tokens = word_tokenize(prompt_lower)
        pos_tagged = pos_tag(tokens)
        
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Extract meaningful nouns and adjectives
        for word, pos in pos_tagged:
            # Skip short words, stopwords, and common words
            if (len(word) >= 3 and 
                word not in stop_words and 
                word.isalpha() and 
                pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']):  # Nouns and adjectives
                
                # Skip very common words that aren't useful as tags
                common_words = {'make', 'create', 'build', 'generate', 'add', 'change', 'with', 'the', 'and', 'for'}
                if word not in common_words:
                    tags.append(word)
    
    except ImportError:
        print("NLTK not available, using basic word extraction")
        # Fallback: extract any meaningful words (3+ characters, alphabetic)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', prompt_lower)
        stop_words = {'the', 'and', 'for', 'with', 'make', 'create', 'build', 'generate', 'add', 'change', 'this', 'that', 'from', 'have', 'will', 'can', 'you', 'are', 'its', 'now', 'but', 'not', 'all', 'one', 'two', 'get', 'see', 'use', 'new', 'way', 'may', 'say', 'each', 'which', 'she', 'how', 'him', 'her', 'his', 'has', 'had'}
        tags.extend([word for word in words if word not in stop_words])
        
    except Exception as e:
        print(f"Warning: Advanced tagging failed, using basic extraction: {e}")
        # Fallback: extract any meaningful words (3+ characters, alphabetic)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', prompt_lower)
        stop_words = {'the', 'and', 'for', 'with', 'make', 'create', 'build', 'generate', 'add', 'change', 'this', 'that', 'from', 'have', 'will', 'can', 'you', 'are', 'its', 'now', 'but', 'not', 'all', 'one', 'two', 'get', 'see', 'use', 'new', 'way', 'may', 'say', 'each', 'which', 'she', 'how', 'him', 'her', 'his', 'has', 'had'}
        tags.extend([word for word in words if word not in stop_words])
    
    # Clean up and deduplicate
    tags = list(set(tags))  # Remove duplicates
    tags = [tag for tag in tags if len(tag) >= 3]  # Keep only meaningful length tags
    
    # Limit to most relevant tags (top 12 to capture more variety)
    return tags[:12]

def add_to_conversation_context(user_prompt: str, generated_prompt: str, memory_key: str):
    """Add interaction to short-term conversation context"""
    interaction = {
        'user_prompt': user_prompt,
        'generated_prompt': generated_prompt,
        'memory_key': memory_key,
        'timestamp': time.time(),
        'tags': extract_tags(user_prompt)
    }
    
    # Add to session context (last 10 interactions)
    memory_manager.session_context.append(interaction)
    
    # Add to current conversation buffer
    memory_manager.conversation_buffer.append(interaction)
    
    # Save session state
    memory_manager.save_session_state()

def get_conversation_context() -> List[Dict]:
    """Get recent conversation context for short-term memory"""
    return list(memory_manager.session_context)

def find_conversation_reference(query: str) -> Optional[Dict]:
    """Find reference in recent conversation context"""
    query_lower = query.lower()
    query_tags = extract_tags(query)
    
    # Search through recent conversations (newest first)
    for interaction in reversed(list(memory_manager.session_context)):
        # Check direct mention in user prompt
        if any(tag in interaction['user_prompt'].lower() for tag in query_tags):
            return interaction
        
        # Check tag overlap
        common_tags = set(query_tags) & set(interaction['tags'])
        if len(common_tags) >= 1:  # At least one common tag
            return interaction
    
    return None

def save_memory_named(name: str, prompt: str, result: bytes, path=MEMORY_PATH):
    """Save to long-term memory with enhanced metadata"""
    memory = load_memory(path)
    
    # Ensure result is base64 encoded
    if isinstance(result, bytes):
        result = base64.b64encode(result).decode('utf-8')
    
    # Generate comprehensive tags
    tags = extract_tags(prompt)
    
    # Create memory entry with timestamp and metadata
    current_time = time.time()
    memory[name.lower()] = {
        "prompt": prompt,
        "result": result,
        "tags": tags,
        "timestamp": current_time,
        "word_count": len(prompt.split()),
        "access_count": 1,
        "last_accessed": current_time
    }
    
    # Update global references
    memory["__last_prompt__"] = name.lower()
    memory["__last_timestamp__"] = current_time
    
    # Save to file
    with open(path, 'w') as f:
        json.dump(memory, f, indent=2)
    
    print(f"💾 Saved to long-term memory '{name}' with tags: {tags}")

def recall_memory_by_name(name: str, path=MEMORY_PATH) -> Optional[bytes]:
    """Recall binary result by name and update access statistics"""
    memory = load_memory(path)
    entry = memory.get(name.lower())
    if entry and "result" in entry:
        # Update access statistics
        entry["access_count"] = entry.get("access_count", 0) + 1
        entry["last_accessed"] = time.time()
        
        # Save updated memory
        with open(path, 'w') as f:
            json.dump(memory, f, indent=2)
        
        try:
            return base64.b64decode(entry["result"])
        except Exception as e:
            print(f"Error decoding memory for '{name}': {e}")
            return None
    return None

def recall_prompt_by_name(name: str, path=MEMORY_PATH) -> Optional[str]:
    """Recall prompt text by name and update access statistics"""
    memory = load_memory(path)
    entry = memory.get(name.lower())
    if entry:
        # Update access statistics
        entry["access_count"] = entry.get("access_count", 0) + 1
        entry["last_accessed"] = time.time()
        
        # Save updated memory
        with open(path, 'w') as f:
            json.dump(memory, f, indent=2)
        
        return entry.get("prompt")
    return None

def recall_last_prompt(path=MEMORY_PATH) -> Optional[str]:
    """Recall the most recent prompt"""
    memory = load_memory(path)
    last_key = memory.get("__last_prompt__")
    if last_key and last_key in memory:
        return memory[last_key].get("prompt")
    return None

def fuzzy_match_name(query: str, path=MEMORY_PATH) -> Optional[str]:
    """Enhanced fuzzy matching with recency and access frequency scoring"""
    memory = load_memory(path)
    query = query.lower().strip()
    
    if not query:
        return None
    
    matches = []
    current_time = time.time()
    
    for key, entry in memory.items():
        if key.startswith("__"):  # Skip metadata keys
            continue
            
        score = 0
        
        # Direct key match (highest score)
        if query == key:
            return key
        
        # Partial key match
        if query in key:
            score += 10
        
        # Key contains query
        if key in query:
            score += 8
        
        # Tag matches
        tags = entry.get("tags", [])
        for tag in tags:
            if query == tag:
                score += 15
            elif query in tag or tag in query:
                score += 5
        
        # Prompt content match
        prompt = entry.get("prompt", "").lower()
        if query in prompt:
            score += 3
        
        # Recency bonus (more recent = higher score)
        timestamp = entry.get("timestamp", 0)
        if timestamp > 0:
            age_hours = (current_time - timestamp) / 3600
            if age_hours < 1:  # Last hour
                score += 5
            elif age_hours < 24:  # Last day
                score += 3
            elif age_hours < 168:  # Last week
                score += 1
        
        # Access frequency bonus
        access_count = entry.get("access_count", 0)
        if access_count > 5:
            score += 2
        elif access_count > 2:
            score += 1
        
        if score > 0:
            matches.append((key, score, timestamp))
    
    # Return best match (by score, then by recency)
    if matches:
        matches.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_match = matches[0]
        print(f"🔍 Fuzzy match for '{query}': found '{best_match[0]}' (score: {best_match[1]})")
        return best_match[0]
    
    return None

def smart_context_retrieval(user_prompt: str) -> Tuple[Optional[str], str]:
    """
    Smart context retrieval that combines short-term and long-term memory
    Returns: (context_prompt, context_source)
    """
    
    # Step 1: Check recent conversation context (SHORT-TERM)
    conversation_ref = find_conversation_reference(user_prompt)
    if conversation_ref:
        print(f"🧠 Found in short-term memory: {conversation_ref['memory_key']}")
        return conversation_ref['generated_prompt'], f"recent conversation ({conversation_ref['memory_key']})"
    
    # Step 2: Check for specific named references (LONG-TERM)
    reference_name = extract_reference_name(user_prompt)
    if reference_name:
        matched_key = fuzzy_match_name(reference_name)
        if matched_key:
            prompt = recall_prompt_by_name(matched_key)
            if prompt:
                print(f"🏛️ Found in long-term memory: {matched_key}")
                return prompt, f"long-term memory ({matched_key})"
    
    # Step 3: Try semantic similarity via vector search (if available)
    try:
        from vector_memory import search_similar_prompt
        similar_entry = search_similar_prompt(user_prompt, top_k=1)
        if similar_entry:
            print(f"🔍 Found via semantic search: {similar_entry['name']}")
            return similar_entry['prompt'], f"semantic similarity ({similar_entry['name']})"
    except ImportError:
        pass
    
    # Step 4: Fall back to last created scene
    last_prompt = recall_last_prompt()
    if last_prompt:
        print("📝 Using last created scene")
        return last_prompt, "last created scene"
    
    return None, "no context found"

def extract_reference_name(prompt: str):
    """Extract reference names from prompt"""
    import re

    # More comprehensive pattern matching
    matches = re.findall(r'\b(cybertruck|bike|robot|city|scene|hoverboard|car|truck|dragon|building|landscape|vehicle|castle|house|tower)\b', prompt.lower())
    return matches[-1] if matches else None

def cleanup_old_memories(max_entries=100, path=MEMORY_PATH):
    """Clean up old memories, keeping frequently accessed and recent ones"""
    memory = load_memory(path)
    
    # Get all entries with scoring
    entries = []
    current_time = time.time()
    
    for key, entry in memory.items():
        if key.startswith("__"):
            continue
        
        # Calculate retention score
        timestamp = entry.get("timestamp", 0)
        access_count = entry.get("access_count", 0)
        last_accessed = entry.get("last_accessed", timestamp)
        
        # Scoring factors
        recency_score = max(0, (7 * 24 * 3600 - (current_time - timestamp)) / (7 * 24 * 3600))  # Recent = higher
        access_score = min(access_count / 10, 1.0)  # Frequently accessed = higher
        recent_access_score = max(0, (24 * 3600 - (current_time - last_accessed)) / (24 * 3600))  # Recently accessed = higher
        
        retention_score = recency_score + access_score + recent_access_score
        
        entries.append((key, retention_score, timestamp))
    
    # If we have too many entries, remove the lowest scoring ones
    if len(entries) > max_entries:
        entries.sort(key=lambda x: x[1], reverse=True)  # Sort by retention score, highest first
        
        # Keep only the best entries
        to_keep = entries[:max_entries]
        to_remove = entries[max_entries:]
        
        # Create new memory dict with only high-scoring entries
        new_memory = {}
        for key, _, _ in to_keep:
            new_memory[key] = memory[key]
        
        # Preserve metadata
        for key in memory:
            if key.startswith("__"):
                new_memory[key] = memory[key]
        
        # Save cleaned memory
        with open(path, 'w') as f:
            json.dump(new_memory, f, indent=2)
        
        print(f"🧹 Cleaned up memory: removed {len(to_remove)} low-priority entries")

def clear_session_memory():
    """Clear short-term session memory (new conversation)"""
    global memory_manager
    memory_manager.session_context.clear()
    memory_manager.conversation_buffer.clear()
    memory_manager.save_session_state()
    print("🔄 Cleared short-term memory for new session")

# Debug functions
def print_memory_status(path=MEMORY_PATH):
    """Print comprehensive memory status"""
    memory = load_memory(path)
    keys = [k for k in memory.keys() if not k.startswith("__")]
    
    print(f"📊 Memory Status:")
    print(f"   Long-term entries: {len(keys)}")
    print(f"   Short-term context: {len(memory_manager.session_context)}")
    print(f"   Last prompt: {memory.get('__last_prompt__', 'None')}")
    print(f"   Recent keys: {', '.join(keys[-5:]) if keys else 'None'}")
    
    return keys

def get_memory_statistics():
    """Get detailed memory statistics"""
    memory = load_memory()
    session_context = get_conversation_context()
    
    stats = {
        "long_term_entries": len([k for k in memory.keys() if not k.startswith("__")]),
        "short_term_entries": len(session_context),
        "total_access_count": sum(entry.get("access_count", 0) for entry in memory.values() if isinstance(entry, dict)),
        "most_accessed": max(memory.items(), key=lambda x: x[1].get("access_count", 0) if isinstance(x[1], dict) else 0, default=("none", {"access_count": 0}))[0]
    }
    
    return stats