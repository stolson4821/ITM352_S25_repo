# Dictionary of 10 multiple choice questions about guitar parts
guitar_questions = {
    1: {
        "question": "What is the part of the guitar that vibrates to produce sound?",
        "options": ["Neck", "Body", "Strings", "Pickups"],
        "answer": "Strings"
    },
    2: {
        "question": "What is the name of the part that holds the guitar's strings in place at the bridge?",
        "options": ["Nut", "Fretboard", "Bridge", "Headstock"],
        "answer": "Bridge"
    },
    3: {
        "question": "Which part of the guitar is used to adjust the pitch of the strings?",
        "options": ["Tuning pegs", "Pickups", "Fretboard", "Strap buttons"],
        "answer": "Tuning pegs"
    },
    4: {
        "question": "What is the function of the guitar's fretboard?",
        "options": ["To amplify sound", "To hold the strings", "To press the strings for notes", "To tune the strings"],
        "answer": "To press the strings for notes"
    },
    5: {
        "question": "What is the headstock of the guitar?",
        "options": ["The neck of the guitar", "The part where the strings are tightened", "The body of the guitar", "The bridge of the guitar"],
        "answer": "The part where the strings are tightened"
    },
    6: {
        "question": "What part of the guitar is responsible for amplifying the sound on an electric guitar?",
        "options": ["Pickups", "Strings", "Fretboard", "Neck"],
        "answer": "Pickups"
    },
    7: {
        "question": "What does the guitar's 'truss rod' do?",
        "options": ["It amplifies the sound", "It adjusts the tension of the strings", "It supports the neck", "It holds the bridge in place"],
        "answer": "It supports the neck"
    },
    8: {
        "question": "What is the purpose of the guitar's bridge?",
        "options": ["To adjust string tension", "To hold the neck in place", "To connect the strings to the body", "To amplify the sound"],
        "answer": "To connect the strings to the body"
    },
    9: {
        "question": "What is the name of the part that allows the player to pick or pluck the strings?",
        "options": ["Pick", "Tuning pegs", "Strap", "Bridge"],
        "answer": "Pick"
    },
    10: {
        "question": "Which part of the guitar is responsible for producing the majority of the sound in an acoustic guitar?",
        "options": ["Neck", "Soundhole", "Strings", "Body"],
        "answer": "Body"
    }
}


#Displaying the questions and answers
for q_num, q_data in guitar_questions.items():
    print(f"Q{q_num}: {q_data['question']}")
    print(f"Options: {', '.join(q_data['options'])}")
    print(f"Answer: {q_data['answer']}\n")

# Save the dictionary as a JSON file
with open('a1questions.json', 'w') as json_file:
    json.dump(guitar_questions, json_file, indent=4)
