Guitar Quiz Application
Features Overview

User Authentication: Secure login system with predefined user credentials
Interactive Quiz: Questions presented one at a time with multiple-choice answers
Score Tracking: User scores are tracked throughout the quiz session
Responsive Design: Studio Ghibli-inspired UI with a clean, modern interface

Technical Components

Backend: Python Flask framework
Frontend: HTML with CSS styling
Data Storage: JSON file for quiz questions, session-based score tracking

Setup Instructions
File Structure
Ensure all files are in their correct locations:
/Assignment_3/
│
├── AS3.py                 # Main Flask application
├── questions.json         # Quiz questions data
├── static/
│   └── style.css          # CSS used AI
│
└── templates/
    ├── index.html         # Landing page
    ├── login.html         # Login form
    ├── ready_to_begin.html # Pre-quiz page
    ├── questions.html     # Question display page
    └── thank_you.html     # Quiz completion page

Start the Server
bashpython AS3.py
or
python3 AS3.py

Access the Application
Open your web browser and navigate to:
http://localhost:5000

Login Credentials
Use one of the following username/password combinations:

Username: port | Password: port123
Username: Teachasst | Password: teachme123
Username: spencer | Password: spencer123
Username: visitor | Password: visit123



User Flow

User arrives at the welcome page
User logs in with credentials
User is welcomed and prompted to start the quiz
Questions are presented one at a time
After answering all questions, user sees their final score

Customization

To Add Questions: Modify the questions.json file to add or change quiz questions
To Make Style Changes: Edit the style.css file to customize the application's appearance
User Management: Add or modify user credentials in the USERS dictionary in AS3.py

Future Enhancements

Database integration for persistent user records
User registration functionality
Score history and leaderboards
Timed quiz mode
Question categories and difficulty levels