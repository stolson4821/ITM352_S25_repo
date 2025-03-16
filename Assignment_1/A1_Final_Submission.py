import json
import random

#Define the file where scores will be stored.
SCORE_FILE = "scores.json"

#Function questions from a JSON file.
#Opens the file and loads it as a JSON object.
def load_questions():
    with open("a1questions.json", "r") as file:
        return json.load(file)

#Function to load existing high scores from the scores file.
#Tries to open and load the data if it exists.
def load_scores():
    try:
        with open(SCORE_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

#Save the current scores to the scores file.
#Opens the scores file in write mode and dumps the scores data as JSON.
def save_scores(scores):
    with open(SCORE_FILE, "w") as file:
        json.dump(scores, file)

#Prompt the user for their username.
#Asks the user to input their username, removes any extra spaces.
def get_user():
    return input("Enter your username: ").strip()

#Function to present a question and handle the user's response.
def ask_question(question_data):
    #Extract the question text and the possible answer options.
    question = question_data["question"]
    options = question_data["options"]
    
    #Ensure the answer is in a list format.
    correct_answers = question_data["answer"] if isinstance(question_data["answer"], list) else [question_data["answer"]]
    
    #Shuffle the answer options to randomize their order.
    random.shuffle(options)
    
    #Display the question to the user.
    print("\n" + question)
    
    #Generate labels (A, B, C, etc.) for each option.
    labels = [chr(65 + i) for i in range(len(options))]
    
    #Create a mapping between the labels (A, B, C) and the answer options.
    option_map = dict(zip(labels, options))
    
    #Display each option to the user with its label (A, B, C).
    for label in labels:
        print(label + ". " + option_map[label])
    
    #Allow the user to select multiple answers, comma-separated (e.g., A, C).
    attempts = 2
    score = 0
    
    #Loop to give the user up to 2 attempts to answer the question correctly.
    while attempts > 0:
        #Prompt the user to select one or more answers and normalize input to uppercase.
        user_input = input("Select the correct answer(s) (comma-separated): ").strip().upper()
        
        #Check if the user input is valid (only contains valid labels).
        selected_labels = user_input.split(",")
        
        #Check that all inputs are valid labels
        if all(label in option_map for label in selected_labels):
            #Convert labels to options and compare with correct answers.
            selected_answers = [option_map[label] for label in selected_labels]
            
            #Compare selected answers to correct answers.
            correct_selections = set(selected_answers) == set(correct_answers)
            
            #If the user selects the correct answers, give a score.
            if correct_selections:
                score = 1 if attempts == 2 else 0.5
                print("Correct!\n")
                return score
            else:
                #Decrease the number of attempts if the answer is wrong.
                attempts -= 1
                #Let the user know how many attempts are remaining.
                if attempts > 0:
                    print(f"Incorrect. Try again. {attempts} attempt(s) left.")
                else:
                    #No attempts left and the answer is still incorrect.
                    print("Incorrect. No attempts left.")
                    score = 0
        else:
            #Handle invalid input (if the user selects something not in the options).
            print("Invalid choice. Please select valid options.")
    #Return the score (0 points if incorrect on both attempts).
    return score

#Main function to run the quiz game.
def quiz_game():
    #Load the saved high scores.
    scores = load_scores()
    # et the username of the player.
    user = get_user()
    #Get the highest score for the user (default to 0 if no score exists).
    user_high_score = scores.get(user, 0)
    #Load the list of questions for the quiz.
    questions = load_questions()
    #Shuffle the order of the questions to randomize the quiz.
    random.shuffle(questions)
    #Initialize the user's total score for this game.
    score = 0

    #Loop through each question and ask it to the user.
    #Add the score for each question to the total score.
    #Display the final score after all questions are answered.
    for question_data in questions:
        score += ask_question(question_data)
    
    print(f"Game Over! Your final score: {score}/{len(questions)}")
    
    #Check if the user achieved a new high score.
    if score > user_high_score:
        print("New high score!")
        scores[user] = score  # Update the user's score in the dictionary.
    
    #Save the updated scores to the file.
    save_scores(scores)
    
    #Determine the grand champion (the user with the highest score).
    grand_champion = max(scores, key=scores.get, default="None")
    
    #Display the grand champion's name and score.
    print(f"Grand Champion: {grand_champion} with {scores.get(grand_champion, 0)} points!")

#Check if this script is being run directly and not imported as a module.
#Run the quiz game if the script is executed directly.
if __name__ == "__main__":
    quiz_game()