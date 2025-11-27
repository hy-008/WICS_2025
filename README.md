# WICS_2025

## Setup:
1. Create a **.env** file in the src folder and add the following:
LLAMACLOUD_API_KEY=your_api_key_here

2. If you haven’t installed dotenv yet, install it: pip install python-dotenv

3. Ensure your Python code loads the .env file:
from dotenv import load_dotenv
load_dotenv()