#!/usr/bin/env python3
"""
Demo script for the Memory Agent system.
This demonstrates the core functionality described in the task specification.
"""

import os
import sys
from app import MemoryAgent

def demo_memory_system():
    """Demonstrate the memory system with the example from the task."""
    
    print("🤖 Memory Agent Demo")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Initialize the memory agent
        print("📡 Initializing Memory Agent...")
        agent = MemoryAgent()
        print("✅ Memory Agent initialized successfully!")
        
        # Demo user ID
        user_id = "demo_user_123"
        
        print(f"\n👤 Using user ID: {user_id}")
        print("\n" + "=" * 50)
        
        # Example 1: Add memories
        print("\n📝 Example 1: Adding Memories")
        print("-" * 30)
        
        message1 = "I use Shram and Magnet as productivity tools"
        print(f"User says: '{message1}'")
        
        result = agent.add_memory(user_id, message1)
        print(f"Result: {result['message']}")
        if result['success']:
            print(f"Facts extracted: {result['facts']}")
        
        print("\n" + "-" * 30)
        
        # Example 2: Delete a memory
        print("\n🗑️  Example 2: Deleting a Memory")
        print("-" * 30)
        
        message2 = "I don't use Magnet anymore"
        print(f"User says: '{message2}'")
        
        result = agent.delete_memory(user_id, message2)
        print(f"Result: {result['message']}")
        if result['success']:
            print(f"Deletion facts: {result['deletion_facts']}")
            print(f"Memories deleted: {result['deleted_count']}")
        
        print("\n" + "-" * 30)
        
        # Example 3: Query with memory
        print("\n❓ Example 3: Querying with Memory")
        print("-" * 30)
        
        question = "What productivity tools do I use?"
        print(f"User asks: '{question}'")
        
        response = agent.query_with_memory(user_id, question)
        print(f"AI Response: {response}")
        
        print("\n" + "-" * 30)
        
        # Example 4: Add more memories
        print("\n📝 Example 4: Adding More Memories")
        print("-" * 30)
        
        message3 = "My name is John and I live in New York"
        print(f"User says: '{message3}'")
        
        result = agent.add_memory(user_id, message3)
        print(f"Result: {result['message']}")
        if result['success']:
            print(f"Facts extracted: {result['facts']}")
        
        print("\n" + "-" * 30)
        
        # Example 5: Query about personal info
        print("\n❓ Example 5: Querying Personal Information")
        print("-" * 30)
        
        question2 = "What's my name and where do I live?"
        print(f"User asks: '{question2}'")
        
        response2 = agent.query_with_memory(user_id, question2)
        print(f"AI Response: {response2}")
        
        print("\n" + "-" * 30)
        
        # Example 6: Memory summary
        print("\n📊 Example 6: Memory Summary")
        print("-" * 30)
        
        summary = agent.get_memory_summary(user_id)
        print(f"Memory Stats: {summary['stats']}")
        print(f"Recent Memories: {len(summary['recent_memories'])} memories")
        
        for i, memory in enumerate(summary['recent_memories'][:3], 1):
            print(f"  {i}. {memory['content']}")
        
        print("\n" + "-" * 30)
        
        # Example 7: Search memories
        print("\n🔍 Example 7: Searching Memories")
        print("-" * 30)
        
        search_query = "productivity tools"
        print(f"Searching for: '{search_query}'")
        
        search_results = agent.search_memories(user_id, search_query, n=3)
        print(f"Found {len(search_results)} relevant memories:")
        
        for i, memory in enumerate(search_results, 1):
            print(f"  {i}. {memory['content']}")
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Fact extraction from natural language")
        print("✅ Memory addition and deletion")
        print("✅ Context-aware responses with GPT")
        print("✅ Memory search and retrieval")
        print("✅ Memory statistics and summaries")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

def interactive_demo():
    """Interactive demo mode."""
    
    print("🤖 Interactive Memory Agent Demo")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        return
    
    try:
        agent = MemoryAgent()
        user_id = "interactive_user"
        
        print("✅ Memory Agent ready!")
        print("\nCommands:")
        print("  add <message>     - Add memories from message")
        print("  delete <message>  - Delete memories from message")
        print("  ask <question>    - Ask a question with memory context")
        print("  search <query>    - Search memories")
        print("  summary           - Show memory summary")
        print("  reset             - Reset all memories")
        print("  quit              - Exit")
        
        while True:
            try:
                command = input(f"\n[{user_id}]> ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'summary':
                    summary = agent.get_memory_summary(user_id)
                    print(f"Memory Stats: {summary['stats']}")
                    for memory in summary['recent_memories'][:5]:
                        print(f"  - {memory['content']}")
                
                elif command.lower() == 'reset':
                    agent.reset_user_memories(user_id)
                    print("✅ All memories reset")
                
                elif command.startswith('add '):
                    message = command[4:].strip()
                    result = agent.add_memory(user_id, message)
                    print(f"Result: {result['message']}")
                    if result['success']:
                        print(f"Facts: {result['facts']}")
                
                elif command.startswith('delete '):
                    message = command[7:].strip()
                    result = agent.delete_memory(user_id, message)
                    print(f"Result: {result['message']}")
                    if result['success']:
                        print(f"Deleted: {result['deleted_count']} memories")
                
                elif command.startswith('ask '):
                    question = command[4:].strip()
                    response = agent.query_with_memory(user_id, question)
                    print(f"AI: {response}")
                
                elif command.startswith('search '):
                    query = command[7:].strip()
                    results = agent.search_memories(user_id, query)
                    print(f"Found {len(results)} memories:")
                    for memory in results:
                        print(f"  - {memory['content']}")
                
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        demo_memory_system()
