import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from translate import Translator
import pyttsx3
import speech_recognition as sr
from PIL import Image
import requests
from gtts import gTTS
import os
from fpdf import FPDF  # For PDF generation
import sqlite3

# Load API keys from .env
load_dotenv()
os.getenv("GOOGLE_API_KEY")
API_KEY = "95f9b9c2c5a5c52e1950302f4d48f13d"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    age INTEGER,
                    profession TEXT,
                    interests TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS admins (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

class UserInfo:
    def init(self):
        self.name = ""
        self.age_description = ""
        self.profession = ""
        self.interests = []

    def set_info(self, name, age, profession, interests):
        self.name = name
        self.age_description = f"{age} years old"
        self.profession = profession
        self.interests = interests

    def get_info(self):
        return {
            "name": self.name,
            "age_description": self.age_description,
            "profession": self.profession,
            "interests": ", ".join(self.interests),
        }
    



# Inject custom CSS for sidebar and UI enhancements
def inject_custom_css():
    st.markdown("""
        <style>
            .css-1aumxhk {
                background-color: #2c3e50 !important;
                padding: 10px;
            }
            .css-qbe2hs {
                font-size: 24px !important;
                color: white !important;
                text-align: center;
            }
            .stButton button {
                width: 100%;
                background-color: #1abc9c;
                color: white;
                padding: 10px 15px;
                border: none;
                font-size: 16px;
                margin: 5px 0;
                border-radius: 8px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .stButton button:hover {
                background-color: #16a085;
            }
            .stImage {
                max-width: 100%;
                border-radius: 8px;
                margin: 10px 0;
            }
            .css-qbe2hs, .css-1gkzq2a, .css-1v3fvcr {
                color: #ecf0f1 !important;
            }
        </style>
    """, unsafe_allow_html=True)

def fetch_weather(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(BASE_URL, params=params)
    
    # Check response status and handle errors
    if response.status_code == 200:
        data = response.json()
        main = data['main']
        weather = data['weather'][0]
        return {
            "temperature": main['temp'],
            "feels_like": main['feels_like'],
            "weather": weather['description'],
            "humidity": main['humidity']
        }
    else:
        # Print response details for debugging
        st.error(f"Error fetching weather data: {response.json().get('message', 'Unknown error')}")
        return None
    
def weather_forecast_page():
    st.header("🌦️ Weather Forecast")
    city = st.text_input("Enter a city name to get the weather forecast:")
    
    if st.button("Get Weather"):
        if city:
            weather_info = fetch_weather(city)
            if weather_info:
                st.write(f"City: {city.capitalize()}")
                st.write(f"Temperature: {weather_info['temperature']}°C")
                st.write(f"Feels Like: {weather_info['feels_like']}°C")
                st.write(f"Weather: {weather_info['weather'].capitalize()}")
                st.write(f"Humidity: {weather_info['humidity']}%")
            else:
                st.error("Could not retrieve weather information. Please check the city name and try again.")
        else:
            st.warning("Please enter a city name.")

# Setup for conversational chatbot
def get_conversational_chain():
    prompt_template = """
    You are a friendly, human-like assistant. Respond to the user's questions in a conversational, casual tone.
    Be helpful and provide insights. If you're unsure about something, admit it politely.
    
    User Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

# Translation function
def translate_text(text, source_language, target_language):
    translator = Translator(from_lang=source_language, to_lang=target_language)
    return translator.translate(text)

# Speech-to-Text function
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand that. Could you say it again?")
        return ""
    except sr.RequestError as e:
        st.write("There was an error:", e)
        return ""

# Text-to-Speech function using gTTS (Google Text-to-Speech)
def text_to_speech(text, language="en"):
    try:
        # Handle languages using gTTS
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")  # Plays the saved audio file (Windows command)
    except Exception as e:
        st.write(f"Error in text-to-speech: {e}")

# Function to fetch real-time data
def fetch_real_time_data(query):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key=AIzaSyBu2yI5jLcgH4VOL8ovG_c1WHOpXWk5YsQ&cx=b0cb544a507a24460"
    response = requests.get(search_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("items", [{}])[0].get("snippet", "No relevant data found.")
    return "Error fetching real-time data."

# Function to process user input with real-time data
def user_input(user_question, conversation_history, source_language, target_language):
    if "latest" in user_question.lower() or "current" in user_question.lower() or "update" in user_question.lower():
        response = fetch_real_time_data(user_question)
    else:
        if source_language != "en":
            user_question = translate_text(user_question, source_language, "en")
        
        chain = get_conversational_chain()
        response = chain.run(question=user_question)
        
        if target_language != "en":
            response = translate_text(response, "en", target_language)
    
    conversation_history.append({"question": user_question, "response": response})
    return response

# Sidebar and navigation
def sidebar_navigation():
    st.sidebar.title("Dashboard")
    if st.sidebar.button("🏠 Home"):
        st.session_state.current_page = "home"
    if st.sidebar.button("🗣️ Chatbot"):
        st.session_state.current_page = "chatbot"
    if st.sidebar.button("🌤️ Weather Forecast"):
        st.session_state.current_page = "weather_forecast_page"
    if st.sidebar.button("🔑 User Login"):
        st.session_state.current_page = "user_login"
    if st.sidebar.button("📝 User Registration"):
        st.session_state.current_page = "user_register"
    if st.sidebar.button("🔒 Admin Login"):
        st.session_state.current_page = "admin_login"
    if st.sidebar.button("📋 Admin Registration"):
        st.session_state.current_page = "admin_register"

def user_login():
    st.subheader("User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()

        if user:
            # Update session state variables
            st.session_state.logged_in = True
            st.session_state.is_admin = False
            st.session_state.user_info = UserInfo()
            st.session_state.user_info.set_info(username, user[2], user[3], user[4].split(',') if user[4] else [])

            # ✅ Redirect to chatbot page
            st.session_state.current_page = "chatbot"
            
            # Welcome notification
            st.success(f"Hi {username}, welcome to the chatbot! 😊")

            # Force a rerun to navigate to the chatbot page
            st.rerun()
        else:
            st.error("Invalid credentials")
        conn.close()


import pandas as pd  # Import pandas for table formatting

def admin_login():
    st.subheader("🔐 Admin Login")
    username = st.text_input("Admin Username")
    password = st.text_input("Admin Password", type="password")

    if st.button("Login as Admin"):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM admins WHERE username = ? AND password = ?", (username, password))
        admin = c.fetchone()

        if admin:
            st.session_state.logged_in = True
            st.session_state.is_admin = True
            st.session_state.current_page = "admin_dashboard"

            # Welcome notification for admins
            st.success(f"✅ Welcome, Admin {username}!")

            # ✅ Fetch all user details
            c.execute("SELECT username, age, profession, interests FROM users")
            users = c.fetchall()

            if users:
                # Convert data to a Pandas DataFrame
                df = pd.DataFrame(users, columns=["Username", "Age", "Profession", "Interests"])
                
                # Option 1: Fill missing values with 0 and convert to int
                df["Age"] = df["Age"].fillna(0).astype(int)
                
                # Alternatively, Option 2: Use pandas' nullable integer type:
                # df["Age"] = df["Age"].astype("Int64")
                
                # Display total users count
                st.subheader(f"📊 Total Users: {len(users)}")
                
                # Display table with better formatting
                st.dataframe(df.style.set_properties(**{'text-align': 'left'}), use_container_width=True)
            else:
                st.warning("🚨 No users found in the database.")
        
        else:
            st.error("❌ Invalid admin credentials")
        
        conn.close()



def user_register():
    st.subheader("User Registration")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    age = st.number_input("Age", min_value=0, max_value=150, step=1)
    profession = st.text_input("Profession")
    interests = st.text_area("Interests (comma-separated)")

    if st.button("Register"):
        if not username or not password:
            st.error("Please fill in all the required fields!")
            return

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, age, profession, interests) VALUES (?, ?, ?, ?, ?)",
                      (username, password, age, profession, interests))
            conn.commit()
            st.success("Registration successful! You can now log in.")
        except sqlite3.IntegrityError:
            st.error("Username already exists. Please choose a different username.")
        finally:
            conn.close()

def admin_register():
    st.subheader("Admin Registration")
    username = st.text_input("Admin Username")
    password = st.text_input("Admin Password", type="password")

    if st.button("Register Admin"):
        if not username or not password:
            st.error("Please fill in all the required fields!")
            return

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO admins (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            st.success("Admin registration successful!")
        except sqlite3.IntegrityError:
            st.error("Admin username already exists. Please choose a different username.")
        finally:
            conn.close()
# Home Page
# Home Page
def welcome_home():
    st.title("Welcome to MINGLEBOT!")
    try:
        image = Image.open("welcome_image.jpeg")
        st.image(image, caption="Hurray!!!", use_container_width=True)
    except FileNotFoundError:
        st.warning("Welcome image not found. Please ensure 'welcome_image.png' is in the correct directory.")
    st.write("Navigate through the app using the sidebar.")

# Chatbot Page
def chatbot_page():
    if not st.session_state.get("logged_in", False):
        st.warning("⚠️ You must be logged in to access this page.")
        return
    st.header("Your Friendly AI Companion 🤖")
    conversation_history = st.session_state.get("conversation_history", [])
    user_question = st.text_input("Ask me anything! Let's chat.")

    col1, col2 = st.columns(2)
    with col1:
        source_language = st.radio("Your Language:", options=["en", "fr", "es", "kn", "hi", "de", "it", "pt", "nl", "ru", "zh", "ar", "ja", "ko", "sv", "tr", "el", "pl", "fi", "cs", "da", "th", "vi"], key="source_language")
    with col2:
        target_language = st.radio("AI Response Language:", options=["en", "fr", "es", "kn", "hi", "de", "it", "pt", "nl", "ru", "zh", "ar", "ja", "ko", "sv", "tr", "el", "pl", "fi", "cs", "da", "th", "vi"], key="target_language")

    if st.button("🎙️ Speak"):
        recognized_text = speech_to_text()
        st.write("You said:", recognized_text)
        user_question = recognized_text
    
    if user_question:
        response = user_input(user_question, conversation_history, source_language, target_language)
        st.write("🤖 Reply: ", response)
    
        if st.button("🔊 Listen"):
            # Pass the target language for TTS
            text_to_speech(response, target_language)
    
    st.subheader("Conversation History")
    for item in conversation_history:
        st.write("🧑 You: ", item["question"])
        st.write("🤖 AI: ", item["response"])
        st.write("---")
    
    st.session_state.conversation_history = conversation_history

    # Button to export chat history to PDF
    if st.button("Export Chat History to PDF"):
        export_chat_history_to_pdf(conversation_history)

# Function to export chat history to PDF
def export_chat_history_to_pdf(conversation_history):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Chat History", ln=True, align='C')
    pdf.ln(10)

    # Adding each conversation history item to the PDF
    for item in conversation_history:
        pdf.cell(200, 10, txt=f"You: {item['question']}", ln=True)
        pdf.cell(200, 10, txt=f"AI: {item['response']}", ln=True)
        pdf.ln(5)
    
    # Save PDF to file
    pdf_output_path = "chat_history.pdf"
    pdf.output(pdf_output_path)
    
    # Provide download link
    with open(pdf_output_path, "rb") as f:
        st.download_button("Download PDF", f, file_name="chat_history.pdf", mime="application/pdf")

# Main function
def main():
    inject_custom_css()
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    sidebar_navigation()
    if st.session_state.current_page == "home":
        welcome_home()
    elif st.session_state.current_page == "chatbot":
        chatbot_page()
    elif st.session_state.current_page == "user_login":
        user_login()
    elif st.session_state.current_page == "user_register":
        user_register()
    elif st.session_state.current_page == "admin_login":
        admin_login()
    elif st.session_state.current_page == "admin_register":
        admin_register()
    elif st.session_state.current_page == "weather_forecast_page":
        weather_forecast_page()

# Run the app
if __name__ == "__main__":
    main()