#!/usr/bin/env python3
"""
Example: How to submit Driver.js tool data to the voice agent

This demonstrates how external tools can inject JSON data into the voice agent's
conversation context, specifically for Driver.js tour data.
"""

import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voice_agent import LocalVoiceAgent

async def example_driverjs_integration():
    """Example of submitting Driver.js tour data"""

    print("🚀 Driver.js Integration Example")
    print("="*50)

    # Create agent instance
    agent = LocalVoiceAgent("demo-room")

    # Enable tools (enabled by default, but good to be explicit)
    agent.enable_tools(True)

    # Simulate Driver.js tour data from a web application
    driverjs_tour_data = {
        "steps": [
            {
                "element": "#welcome-section",
                "popover": {
                    "title": "Welcome to our app!",
                    "description": "This is the main dashboard where you can see all your activities."
                }
            },
            {
                "element": "#features-panel",
                "popover": {
                    "title": "Features Panel",
                    "description": "Here you can access all the powerful features of our application."
                }
            },
            {
                "element": "#settings-button",
                "popover": {
                    "title": "Settings",
                    "description": "Customize your experience and preferences here."
                }
            }
        ],
        "showProgress": True,
        "showButtons": ["next", "previous", "close"],
        "onDestroyed": lambda: print("Tour completed!")
    }

    # User ID - this would come from your authentication system
    user_id = "user-12345"

    print(f"📥 Submitting Driver.js tour data for user: {user_id}")

    # Submit the Driver.js data to the voice agent
    success = await agent.submit_tool_data(
        tool_name="driverjs",
        data=driverjs_tour_data,
        user_id=user_id
    )

    if success:
        print("✅ Driver.js data successfully submitted to voice agent!")
        print("\n💬 Now when the user asks about the tour, the voice agent will have context:")
        print("   User: 'What was in that tour I just saw?'")
        print("   Agent: 'The tour had 3 steps: Welcome, Features Panel, and Settings...'")

        # Show what was stored in the conversation history
        if user_id in agent.user_conversation_history:
            conversation = agent.user_conversation_history[user_id]
            print(f"\n📋 Conversation history now contains {len(conversation)} messages")
            print("   Last message content preview:")
            last_msg = conversation[-1]["content"]
            print(f"   {last_msg[:100]}...")
    else:
        print("❌ Failed to submit Driver.js data")

    return success

async def example_custom_tool_integration():
    """Example of submitting custom tool data"""

    print("\n\n🔧 Custom Tool Integration Example")
    print("="*50)

    # Create agent instance
    agent = LocalVoiceAgent("demo-room")

    # Example: Data from a custom analytics tool
    custom_tool_data = {
        "user_activity": {
            "last_login": "2024-01-15T10:30:00Z",
            "features_used": ["dashboard", "reports", "export"],
            "session_duration": "23 minutes"
        },
        "recommendations": [
            "You might want to try the new data visualization feature",
            "Consider setting up automated reports"
        ]
    }

    user_id = "user-67890"

    print(f"📥 Submitting custom tool data for user: {user_id}")

    # Submit custom tool data
    success = await agent.submit_tool_data(
        tool_name="analytics",
        data=custom_tool_data,
        user_id=user_id
    )

    if success:
        print("✅ Custom tool data successfully submitted!")
        print("\n💬 Now the voice agent can use this context:")
        print("   User: 'What can you tell me about my usage?'")
        print("   Agent: 'Based on your activity, you used dashboard, reports...'")

    return success

async def main():
    """Run all examples"""

    print("Driver.js and Tool Integration Examples")
    print("="*60)
    print("This demonstrates how to submit external tool data")
    print("to the voice agent for contextual conversations.\n")

    # Run examples
    success1 = await example_driverjs_integration()
    success2 = await example_custom_tool_integration()

    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 All examples completed successfully!")
        print("\nKey Takeaways:")
        print("1. Use submit_tool_data() to inject JSON data")
        print("2. Specify tool_name='driverjs' for Driver.js data")
        print("3. Provide user_id to maintain per-user context")
        print("4. The agent automatically analyzes and stores the data")
        print("5. Future conversations will have access to this context")
    else:
        print("⚠️ Some examples failed")

    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
