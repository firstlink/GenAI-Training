"""
Lab 1 - Setup Test
Solution for verifying API key configuration
"""

from dotenv import load_dotenv
import os

load_dotenv()

# Check API keys
openai_key = os.getenv('OPENAI_API_KEY')
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
google_key = os.getenv('GOOGLE_API_KEY')

print("=" * 60)
print("API KEY CONFIGURATION TEST")
print("=" * 60)
print("\nAPI Key Status:")
print(f"✅ OpenAI: {'Loaded (' + openai_key[:10] + '...' + ')' if openai_key else '❌ Missing'}")
print(f"✅ Anthropic: {'Loaded (' + anthropic_key[:10] + '...' + ')' if anthropic_key else '❌ Missing'}")
print(f"✅ Google: {'Loaded (' + google_key[:10] + '...' + ')' if google_key else '❌ Missing'}")

# You need at least ONE key to proceed
if openai_key or anthropic_key or google_key:
    print("\n🎉 Setup complete! You're ready to code.")
    print("\nRecommendation: Use OpenAI for most exercises (best documented)")
else:
    print("\n⚠️ No API keys found. Please add at least one key to .env file.")
    print("\nCreate a .env file with:")
    print("OPENAI_API_KEY=sk-your-key-here")
    print("ANTHROPIC_API_KEY=sk-ant-your-key-here")
    print("GOOGLE_API_KEY=your-key-here")
